import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ShakeDrop import ShakeDrop
from contextlib import contextmanager


class PyramidBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, 
                 stride=1, padding=1, shakedrop=False, p_shakedrop=1.0):
        super(PyramidBlock, self).__init__()
        self.branch = nn.Sequential(nn.BatchNorm2d(in_channel),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                                              padding=padding, stride=stride, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channel, out_channel, kernel_size=3, 
                                              padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(out_channel))
        self.downsample = stride == 2
        self.shortcut   = nn.AvgPool2d(2) if self.downsample else None
        
        self.shakedrop  = shakedrop
        if self.shakedrop:
            self.shakedrop_layer = ShakeDrop(p_drop=p_shakedrop, alpha=[-1,1])
        
    def forward(self, x):
        #print('downsample', self.downsample)
        #print('input', x.shape)
        h0 = self.shortcut(x) if self.downsample else x
        #print('h0',h0.shape)
        h  = self.branch(x)
        #print('h', h.shape)
        h  = self.shakedrop_layer(h) if self.shakedrop else h
        #print('h', h.shape)
        
        ###padding zero is enough for pyramidNet
        pad_zero = torch.autograd.Variable(torch.zeros(h0.size(0),
                                                       h.size(1)-h0.size(1),
                                                       h0.size(2),
                                                       h0.size(3)).float().to(x.device))
        #print('pad_zero', pad_zero.shape)
        h0  = torch.cat([h0, pad_zero], dim=1)
        #print('h0', h0.shape)
        return h+h0        
    
class PyramidNet_ShakeDrop(nn.Module):
    
    def __init__(self, conv1_in_channel=9, depth=20, alpha=270, num_class=6,
                 shakedrop=False, block=PyramidBlock):
        super(PyramidNet_ShakeDrop, self).__init__()
        
        if (depth-2) % 6 != 0:
            raise ValueError('depth should be one of 20, 32, 44, 56, 110, 1202')
        
        self.in_ch     = 16
        self.shakedrop = shakedrop
        
        # PyramidNet
        n_units = (depth - 2) // 6
        channel = lambda x: math.ceil( alpha * (x+1) / (3 * n_units) )
        self.in_chs  = [self.in_ch] + [self.in_ch + channel(i) for i in range (n_units * 3)]
        #print(self.in_chs)
        
        # Stochastic Depth
        self.p_L = 0.5
        linear_decay = lambda x: (1-self.p_L) * x / (3*n_units)
        self.ps_shakedrop = [linear_decay(i) for i in range(3*n_units)]
        
        self.u_idx   = 0
        
        ### Model ###
        
        # input shape (batch, 3, 522, 19)
        
        self.conv1   = nn.Conv2d(conv1_in_channel, self.in_chs[0], kernel_size=(7,7),
                               stride=(2,2), padding=(3,3), bias=False)
        self.bn1     = nn.BatchNorm2d(self.in_chs[0])
        self.relu1   = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1  = self._make_layer(n_units, block, 1, (1,1))
        self.layer2  = self._make_layer(n_units, block, 2, (1,0))
        self.layer3  = self._make_layer(n_units, block, 2, (1,1))
        
        self.bn_out  = nn.BatchNorm2d(self.in_chs[-1])
        self.relu_out= nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=(11,1))
        self.fc_out  = nn.Linear(self.in_chs[-1], num_class)
        
        #output shape (batch, 6)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # MSRA initializer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        #print('1',x.shape)
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        #print('2',h.shape)
        h = self.layer1(h)
        #print('3',h.shape)
        #print('==================================================')
        h = self.layer2(h)
        #print('4',h.shape)
        #print('==================================================')
        h = self.layer3(h)
        #print('5',h.shape)
        #print('==================================================')
        h = self.relu_out(self.bn_out(h))
        #print('6',h.shape)
        h = self.avgpool(h)
        #print('7',h.shape)
        h = h.view(h.size(0), -1)
        #print('8',h.shape)
        h = self.fc_out(h)
        #print('9',h.shape)
        return h
    
    def _make_layer(self, n_units, block, stride=1, padding=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1],
                                stride, padding, self.shakedrop, self.ps_shakedrop[self.u_idx]))
            self.u_idx += 1
            stride = 1
            padding = 1
        return nn.Sequential(*layers)
    
    def run_on_batch(self, batch, batch_ul=None, VAT=False, tech_weights=None, per_song_loader=None):
    #   criterion = nn.CrossEntropyLoss(weight=tech_weights, reduction='mean')
      criterion = nn.CrossEntropyLoss(reduction='mean')
      vat_loss = VATLoss()
    
      spec = batch['feature']
      out = self(spec)
      loss = criterion(out, batch['label'])
      if VAT:
        ### [TODO]  Change unsupervised function here
        ul_criterion = nn.CrossEntropyLoss(reduction='mean')

        spec = batch_ul['feature']
        loss_ul = vat_loss(self, spec)

        ### [TODO]  Change unsupervised function here
        pseu_label = out.argmax(1)
        loss_ul = ul_criterion(out, pseu_label)
        loss += loss_ul

      predictions = {
                'technique_pred': out,
                'annotation': batch['label']
      }
      losses = {
                'super_loss': loss
      }

      return predictions, losses, spec

### VAT for unlabelled data ###
@contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= (torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8)
    return d

class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=40.0, ip=2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
            # pred = F.softmax(model(x).view(3,-1,2), dim=2).view(-1,6)
            
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.softmax(pred_hat, dim=1)
                # logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.softmax(pred_hat, dim=1)
            # logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
