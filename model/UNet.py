import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import librosa
from nnAudio import features
from .constants import *
from .utils import Normalization

class MutliHeadAttention1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, groups=1, position=True, bias=False):
        """kernel_size is the 1D local attention window size"""

        super().__init__()
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.position = position
        
        # Padding should always be (kernel_size-1)/2
        # Isn't it?
        self.padding = (kernel_size-1)//2
        self.groups = groups

        # Make sure the feature dim is divisible by the n_heads
        assert self.out_features % self.groups == 0, f"out_channels should be divided by groups. (example: out_channels: 40, groups: 4). Now out_channels={self.out_features}, groups={self.groups}"
        assert (kernel_size-1) % 2 == 0, "kernal size must be odd number"

        if self.position:
            # Relative position encoding
            self.rel = nn.Parameter(torch.randn(1, out_features, kernel_size), requires_grad=True)

        # Input shape = (batch, len, feat)
        
        # Increasing the channel depth (feature dim) with Conv2D
        # kernel_size=1 such that it expands only the feature dim
        # without affecting other dimensions
        self.W_k = nn.Linear(in_features, out_features, bias=bias)
        self.W_q = nn.Linear(in_features, out_features, bias=bias)
        self.W_v = nn.Linear(in_features, out_features, bias=bias)

        self.reset_parameters()

    def forward(self, x):

        batch, seq_len, feat_dim = x.size()

        padded_x = F.pad(x, [0, 0, self.padding, self.padding])
        q_out = self.W_q(x)
        k_out = self.W_k(padded_x)
        v_out = self.W_v(padded_x)
        k_out = k_out.unfold(1, self.kernel_size, self.stride)
        # (batch, L, feature, local_window)
        
        v_out = v_out.unfold(1, self.kernel_size, self.stride)
        # (batch, L, feature, local_window)
        
        if self.position:
            k_out = k_out + self.rel # relative position?

        k_out = k_out.contiguous().view(batch, seq_len, self.groups, self.out_features // self.groups, -1)

        v_out = v_out.contiguous().view(batch, seq_len, self.groups, self.out_features // self.groups, -1)
        # (batch, L, n_heads, feature_per_head, local_window)
        
        # expand the last dimension s.t. it can multiple with the local att window
        q_out = q_out.view(batch, seq_len, self.groups, self.out_features // self.groups, 1)
        # (batch, L, n_heads, feature_per_head, 1)
        
        energy = (q_out * k_out).sum(-2, keepdim=True)
        
        attention = F.softmax(energy, dim=-1)
        # (batch, L, n_heads, 1, local_window)
        
        out = attention*v_out
#         out = torch.einsum('blnhk,blnhk -> blnh', attention, v_out).view(batch, seq_len, -1)
        # (batch, c, H, W)
        
        return out.sum(-1).flatten(2), attention.squeeze(3)

    def reset_parameters(self):
        init.kaiming_normal_(self.W_k.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.W_v.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.W_q.weight, mode='fan_out', nonlinearity='relu')
        if self.position:
            init.normal_(self.rel, 0, 1)

class Stack(nn.Module):
    def __init__(self, input_size, hidden_dim, attn_size=31, attn_group=4, output_dim=88, dropout=0.3):
        super().__init__() 
        self.attention = MutliHeadAttention1D(input_size, hidden_dim, attn_size, position=True, groups=attn_group)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)        
        
    def forward(self, x):
        x, a = self.attention(x)
        x = self.linear(x)
        x = self.dropout(x)
        # stack shape: (8, 233, 10)
        return x, a
        
''' BN for block '''
batchNorm_momentum = 0.1
### U-net for the labelled data ###
''' Encoder '''
class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride, drop=None):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)
        self.drop = drop
        if drop is not None: 
            self.dropout = nn.Dropout(drop)
        # self.maxpool = nn.MaxPool2d(kernel_size=ds_ksize, stride=ds_stride)
        # self.pool = pool

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
            # if self.pool:
            #     xp = self.maxpool(xp)
        # elif self.pool:
        #     xp = self.maxpool(x12)
        x12 += self.skip(x)
        if self.drop is not None:
            x12 = self.dropout(x12)
        xp = self.ds(x12)


        return xp, xp, x12.size()

class Encoder1(nn.Module):
    def __init__(self,inp,ds_ksize, ds_stride, drop=None):
        super(Encoder1, self).__init__()
        self.block1 = block(inp,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride, drop=drop)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 
        # self.drop = drop
        # if self.drop is not None:
        #     self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        # if self.drop is not None and self.training:
        #     x3 = self.dropout(x3)
        x4,idx4,s4 = self.block4(x3)
        # if self.drop is not None:
        #     x4 = self.dropout(x4)    
        skip3=self.conv1(x3) 
        skip2=self.conv2(x2) 
        skip1=self.conv3(x1) 
        return x4,[s4,s3,s2,s1],[skip3,skip2,skip1,x]

class Encoder2(nn.Module):
    def __init__(self,inp,ds_ksize, ds_stride, drop=None):
        super(Encoder2, self).__init__()
        self.block1 = block(inp,16,(3,3),(1,1),ds_ksize, ds_stride)
        self.block2 = block(16,32,(3,3),(1,1),ds_ksize, ds_stride)
        self.block3 = block(32,64,(3,3),(1,1),ds_ksize, ds_stride)
        self.block4 = block(64,128,(3,3),(1,1),ds_ksize, ds_stride)

        self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
        self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 
        self.drop = drop
        if self.drop is not None:
            self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        if self.drop is not None:
            x3 = self.dropout(x3)
        x4,idx4,s4 = self.block4(x3)
        if self.drop is not None:
            x4 = self.dropout(x4)    

        skip3=self.conv1(x3) 
        skip2=self.conv2(x2) 
        skip1=self.conv3(x1) 

        return x4,[s4,s3,s2,s1],[skip3,skip2,skip1,x]
''' Decoder '''
class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride, skip, drop=None):
        super(d_block, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn1= nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv2 = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)

        if not isLast:
            # self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            if skip:
                self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride)
            else:
                self.us = nn.ConvTranspose2d(inp-out, inp, kernel_size=ds_ksize, stride=ds_stride)
        else:
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride)
        self.drop = drop
        if drop is not None: 
            self.dropout = nn.Dropout(drop)

    def forward(self, x, size=None, isLast=None, skip=None):
        # up sampling
        x = self.us(x,output_size=size)

        if not isLast: 
            # if x.shape[-1] < size[-1]:
            #     x = F.pad(x, (0, 1), "constant", 0)

            if skip is not None:
                # pad odd to even
                # skip connection
                x = torch.cat((x, skip), 1)

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        if self.drop is not None:
            x = self.dropout(x)
        if isLast:
            x = self.conv2(x)
        else:
            x = F.leaky_relu(self.bn2(self.conv2(x)))
        return x


class Decoder1(nn.Module):
    def __init__(self,ds_ksize, ds_stride, num_output, skip=True, drop=None): # num_techniques?
        super(Decoder1, self).__init__()
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.d_block4 = d_block(16,num_output,True,(3,3),(1,1),ds_ksize, ds_stride, skip, drop=drop)
        # self.drop = drop
        # if self.drop is not None:
        #     self.dropout = nn.Dropout(drop)

    def forward(self, x, s, c=[None,None,None,None,None]):
        x = self.d_block1(x,s[0],False,c[0])
        x = self.d_block2(x,s[1],False,c[1])       
        x = self.d_block3(x,s[2],False,c[2])
        # if self.drop is not None and self.training:
        #     x = self.dropout(x)        
        x = self.d_block4(x,s[3],True,c[3])
        # if self.drop is not None:
        #     x = self.dropout(x) 

        return x

class Decoder2(nn.Module):
    def __init__(self,ds_ksize, ds_stride, num_output, skip=True, drop=None): # num_techniques?
        super(Decoder2, self).__init__()
        self.d_block1 = d_block(192,64,False,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.d_block2 = d_block(96,32,False,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.d_block3 = d_block(48,16,False,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.d_block4 = d_block(16,num_output,True,(3,3),(1,1),ds_ksize, ds_stride, skip)
        self.drop = drop
        if self.drop is not None:
            self.dropout = nn.Dropout(drop)

    def forward(self, x, s, c=[None,None,None,None]):
        x = self.d_block1(x,s[0],False,c[0])
        x = self.d_block2(x,s[1],False,c[1])       
        x = self.d_block3(x,s[2],False,c[2])
        if self.drop is not None and self.training:
            x = self.dropout(x)        
        x = self.d_block4(x,s[3],True,c[3])
        if self.drop is not None:
            x = self.dropout(x) 

        return x # This is required to boost the accuracy

class Spec2Roll(nn.Module):
    def __init__(self, ds_ksize, ds_stride, complexity=4):
        super().__init__() 
        ##### tech SOTA setting #####
        self.state_group_encoder = Encoder2(4, ds_ksize, ds_stride, drop=0.2)
        self.state_decoder = Decoder2(ds_ksize, ds_stride, 1, drop=0.3)#, skip=False)
        self.group_decoder = Decoder2(ds_ksize, ds_stride, 1, drop=0.3)
        self.note_tech_encoder = Encoder2(6, ds_ksize, ds_stride, drop=0.1)
        self.note_decoder = Decoder2(ds_ksize, ds_stride, 1, drop=0.3)
        self.tech_decoder = Decoder2(ds_ksize, ds_stride, 1, drop=0.3)

        self.state_attention = Stack(input_size=N_BINS, hidden_dim=768, attn_size=31, attn_group=6, output_dim=3, dropout=0.2)
        self.group_attention = Stack(input_size=N_BINS, hidden_dim=768, attn_size=31, attn_group=6, output_dim=4, dropout=0.2)
        self.note_attention = Stack(input_size=N_BINS, hidden_dim=768, attn_size=31, attn_group=6, output_dim=50, dropout=0.2)
        self.tech_attention = Stack(input_size=N_BINS, hidden_dim=768, attn_size=31, attn_group=6, output_dim=10, dropout=0.2)
        self.state_note_attention = Stack(input_size=53, hidden_dim=768, attn_size=31, attn_group=6, output_dim=50, dropout=0.2)
        self.group_tech_attention = Stack(input_size=64, hidden_dim=768, attn_size=31, attn_group=6, output_dim=10, dropout=0.2)

        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, training):
        # state note U-net
        state_group_enc,s,c = self.state_group_encoder(x)
        state_post = self.state_decoder(state_group_enc,s,c)
        group_post = self.group_decoder(state_group_enc,s,c)
        # group tech U-net
        x1 = torch.cat((state_post.detach(), group_post.detach(), x), 1)
        # x1 = torch.cat((state_post.detach(), x), 1)
        note_tech_enc,s,c = self.note_tech_encoder(x1)
        note_post = self.note_decoder(note_tech_enc,s,c)
        tech_post = self.tech_decoder(note_tech_enc,s,c)

        #################### attention #########################
        state_post_a, a = self.state_attention(state_post.squeeze(1))
        group_post_a, a = self.group_attention(group_post.squeeze(1))
        note_post_ab, a = self.note_attention(note_post.squeeze(1))
        tech_post_ab, a = self.tech_attention(tech_post.squeeze(1))
        
        state_prob = torch.sigmoid(state_post_a.detach())
        group_prob = torch.sigmoid(group_post_a.detach())
        note_prob = torch.sigmoid(note_post_ab)
        tech_prob = torch.sigmoid(tech_post_ab)

        note_prob_tech = note_prob.detach()

        state_note_cat = torch.cat((state_prob.squeeze(1), note_prob.squeeze(1)), 2)
        tech_cat = torch.cat((note_prob_tech.squeeze(1), group_prob.squeeze(1), tech_prob.squeeze(1)), 2)

        note_post_a, a = self.state_note_attention(state_note_cat)
        tech_post_a, a = self.group_tech_attention(tech_cat)

        return (state_post_a, group_post_a, note_post_a, tech_post_a, note_post_ab, tech_post_ab), (state_post.squeeze(1), group_post.squeeze(1), note_post.squeeze(1), tech_post.squeeze(1)), (state_group_enc, note_tech_enc)

### VAT for unlabelled data ###
''' loss for VAT '''
def _l2_normalize(d, binwise):
    # input shape (batch, timesteps, bins, ?)
    if binwise==True:
        d = d/(torch.abs(d)+1e-8)
    else:
        d = d/(torch.norm(d, dim=-1, keepdim=True))
    return d

def binary_kl_div(y_pred, y_ref):
    y_pred = torch.clamp(y_pred, 1e-4, 0.9999) # prevent inf in kl_div
    y_ref = torch.clamp(y_ref, 1e-4, 0.9999)
    q = torch.stack((y_pred, 1-y_pred), -1)
    p = torch.stack((y_ref, 1-y_ref), -1) 
    assert torch.isnan(p.log()).any()==False, "r_adv exploded, please debug tune down the XI for VAT"
    assert torch.isinf(p.log()).any()==False, "r_adv vanished, please debug tune up the XI for VAT"
    return F.kl_div(p.log(), q, reduction='batchmean')   
                   
''' VAT's body '''
class UNet_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """
    def __init__(self, XI, epsilon, n_power, KL_Div, reconstruction=False, weights=None):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon
        self.KL_Div = KL_Div
        
        self.binwise = False
        self.reconstruction = reconstruction

    def forward(self, model, x):  
        with torch.no_grad():
            y_ref, _ = model.transcriber(x) # This will be used as a label, therefore no need grad()
            
#             if self.reconstruction:
#                 technique, _ = model.transcriber(x)
#                 reconstruction, _ = self.reconstructor(technique)
#                 technique2_ref, _ = self.transcriber(reconstruction)                
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
#         if self.reconstruction:
#             d2 = torch.randn_like(x, requires_grad=True) # Need gradient            
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d, binwise=self.binwise)
            x_adv = (x + r).clamp(0,1)
            y_pred, _ = model.transcriber(x_adv)
            if self.KL_Div==True:
                loss = binary_kl_div(y_pred, y_ref)
            else:
                loss = F.binary_cross_entropy(y_pred, y_ref)
                # loss = self.criterion(y_pred, y_ref)
            loss.backward() # Calculate gradient wrt d
            d = d.grad.detach()*1e10
            model.zero_grad() # prevent gradient change in the model 

        # generating virtual labels and calculate VAT
        # normalise the adversarial vector 𝑟adv along the timestep dimension 
        r_adv = self.epsilon * _l2_normalize(d, binwise=self.binwise)
        assert torch.isnan(r_adv).any()==False, f"r_adv has nan, d min={d.min()} d max={d.max()} d mean={d.mean()} please debug tune down the XI for VAT"
        assert torch.isnan(r_adv).any()==False, f"r_adv has inf, d min={d.min()} d max={d.max()} d mean={d.mean()} please debug tune down the XI for VAT"
#         print(f'd max = {d.max()}\td min = {d.min()}')
#         print(f'r_adv max = {r_adv.max()}\tr_adv min = {r_adv.min()}')        
#         logit_p = logit.detach()
        x_adv = (x + r_adv).clamp(0,1)
        y_pred, _ = model.transcriber(x_adv)
        
        if self.KL_Div==True:
            vat_loss = binary_kl_div(y_pred, y_ref)          
        else:
            vat_loss = F.binary_cross_entropy(y_pred, y_ref)              

        return vat_loss, r_adv, _l2_normalize(d, binwise=self.binwise)  # already averaged   

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=None, class_num=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target, weight=None):
        eps = 1e-12
        if self.label_smooth is not None:
            probs = F.log_softmax(pred, dim=1) # softmax + log
            target = torch.clamp(
                target.float(),
                min=self.label_smooth/(self.class_num-1),
                max=1.0 - self.label_smooth)
            if weight is not None:
                loss = -1 * torch.sum((target * probs) * weight, 1)
            else:
                loss = -1 * torch.sum(target * probs, 1)
            return loss.mean()
        else:
            # loss = -1 * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)

### Implement U-net for techniques detection ###
class UNet(nn.Module):
    def __init__(self, ds_ksize, ds_stride, log=True, reconstruction=True, mode='imagewise', spec='Mel', device='cpu', XI=1e-6, eps=1e-2, weights=None):
        super().__init__()
        global N_BINS # using the N_BINS parameter from constant.py
        
        # Selecting the type of spectrogram to use
        if spec == 'CQT':
            r=2
            N_BINS = 88*r
            self.spectrogram = features.CQT1992v2(sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                      n_bins=N_BINS, fmin=27.5, 
                                                      bins_per_octave=12*r, trainable=False)
        elif spec == 'Mel':
            self.spectrogram_1 = features.MelSpectrogram(sr=SAMPLE_RATE, n_fft=512, n_mels=N_BINS,
                                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                          trainable_mel=False, trainable_STFT=False)
            self.spectrogram_2 = features.MelSpectrogram(sr=SAMPLE_RATE, n_fft=768, n_mels=N_BINS,
                                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                          trainable_mel=False, trainable_STFT=False)
            self.spectrogram_3 = features.MelSpectrogram(sr=SAMPLE_RATE, n_fft=1024, n_mels=N_BINS,
                                                          hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX,
                                                          trainable_mel=False, trainable_STFT=False)
        elif spec == 'CFP':
            self.spectrogram = features.CFP(fs=SAMPLE_RATE,
                                               fr=4,
                                               window_size=WINDOW_LENGTH,
                                               hop_length=HOP_LENGTH,
                                               fc=MEL_FMIN,
                                               tc=1/MEL_FMAX)       
            N_BINS = self.spectrogram.quef2logfreq_matrix.shape[0]
        # TODO: add S, GC, GCoS here
        else:
            print(f'Please select a correct spectrogram')                

        self.log = log
        self.normalize = Normalization(mode)          
        self.reconstruction = reconstruction
        self.vat_loss = UNet_VAT(XI, eps, 1, False, weights)        
        self.device = device   
        self.transcriber = Spec2Roll(ds_ksize, ds_stride)
        self.weights = weights

    def get_spec(self, audio, n_fft):
        audio = audio.cpu().numpy()
        mel = librosa.feature.melspectrogram(
            audio,
            sr=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=HOP_LENGTH,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            n_mels=N_BINS).astype(np.float32)
        
        return torch.from_numpy(mel).cuda()

    def forward(self, x):
        # U-net 1
        pred, post, latent = self.transcriber(x, self.training)
        return pred, post, latent

    def run_on_batch(self, batch, batch_ul=None, VAT=False):
        device = self.device
        audio = batch['audio']

        if batch['label'].dim() == 2:
            label = batch['label'].unsqueeze(0)
        else:
            label = batch['label'] # tech + note

        state_label = label[:, :, :3]
        group_label = label[:, :, 3:7]
        note_label = label[:, :, 7:57]
        tech_label = label[:, :, 57:]

        # technique = batch['technique'].flatten().type(torch.LongTensor).to(device)
        # use the weight for the unbalanced tech label
        if self.weights is not None:
            state_weights = self.weights[0]
            group_weights = self.weights[1]
            tech_weights = self.weights[2]
        #####################for unlabelled audio###########################
        # do VAT for unlabelled audio
        if batch_ul:
            audio_ul = batch_ul['audio']
            if audio_ul.dim() == 2 and audio_ul.shape[-1] != 2:
                # audio_ul is already mono
                audio_ul.unsqueeze_(-1)
                audio_ul = audio_ul[:, :, 0]

            # WARNING: change the input channel from 1 to 2 to test hop 256 and 512
            spec_1 = self.spectrogram_1(audio_ul) # x = torch.rand(8,229, 640)
            spec_2 = self.spectrogram_2(audio_ul)
            spec_3 = self.spectrogram_3(audio_ul)
            if self.log:
                spec_1 = torch.log(1 + 30 * spec_1) # (spec_1 + 1e-5)
                spec_2 = torch.log(1 + 30 * spec_2)
                spec_3 = torch.log(1 + 30 * spec_3)
                spec_1 = self.normalize.transform(spec_1)
                spec_2 = self.normalize.transform(spec_2)
                spec_3 = self.normalize.transform(spec_3)

            spec_1 = spec_1.transpose(-1,-2).unsqueeze(1) # torch.rand(8, 1, 229, 640)
            spec_2 = spec_2.transpose(-1,-2).unsqueeze(1)
            spec_3 = spec_3.transpose(-1,-2).unsqueeze(1)
            
            spec = torch.cat((spec_1, spec_2, spec_3), 1) # torch.rand(8, 2, 229, 640)
            lds_ul, _, r_norm_ul = self.vat_loss(self, spec)
        else:
            # lds_ul = torch.tensor(0.)
            lds_ul = torch.tensor(0.)
            r_norm_ul = torch.tensor(0.)
        #####################for unlabelled audio###########################
        # Converting audio to spectrograms
        # spectrogram needs input (num_audio, len_audio):
        ## convert each batch to a single channel audio
        if audio.shape[-1] == 2:
            # stereo
            if audio.dim() == 2:
                # validation audio
                audio = audio.unsqueeze(0)
            if audio.dim() == 3:
                # batch 
                audio = audio[:, :, 0]

        # spec_1 = self.get_spec(audio, 512)
        spec_1 = self.spectrogram_1(audio)
        spec_2 = self.spectrogram_2(audio)
        spec_3 = self.spectrogram_3(audio)
        # log compression
        if self.log:
            spec_1 = torch.log(1 + 10 * spec_1) # (spec_1 + 1e-5)
            spec_2 = torch.log(1 + 10 * spec_2)
            spec_3 = torch.log(1 + 10 * spec_3) # 1 + 5 * spec_3
        # Normalizing spectrograms
        spec_1 = self.normalize.transform(spec_1)
        spec_2 = self.normalize.transform(spec_2)
        spec_3 = self.normalize.transform(spec_3)
        # swap spec bins with timesteps so that it fits attention later 
        spec_1 = spec_1.transpose(-1,-2).unsqueeze(1) # shape (8,1,640,229)
        spec_2 = spec_2.transpose(-1,-2).unsqueeze(1)
        spec_3 = spec_3.transpose(-1,-2).unsqueeze(1)
        ############ spectral flux ############
        spec_flux = spec_2 - torch.cat((spec_2[:,:,1:,:], torch.zeros(spec_2[:,:,1,:].unsqueeze(2).shape).cuda()), 2)
        ############ spectral flux ############
        spec = torch.cat((spec_1, spec_2, spec_3, spec_flux), 1)
        # do VAT for labelled audio
        if VAT:
            lds_l, r_adv, r_norm_l = self.vat_loss(self, spec)
            r_adv = r_adv.squeeze(1) # remove the channel dimension
        else:
            r_adv = None
            lds_l = torch.tensor(0.)
            r_norm_l = torch.tensor(0.)
        
        
        pred, post, latent = self(spec)
        # state_criterion = CrossEntropyLoss(0.05, 50)
        # softmax = nn.Softmax(dim=-1)
        if self.training:
            predictions = {
                    'note_state': pred[0].reshape(-1, 3),
                    'tech_group': pred[1].reshape(-1, 4),
                    'note': pred[2].reshape(-1, 50),
                    'tech': pred[3].reshape(-1, 10),
                    'r_adv': r_adv,
                    }
            losses = {
                    'loss/train_state': 3 * F.binary_cross_entropy_with_logits(pred[0], state_label, pos_weight=state_weights),
                    'loss/train_group': 3 * F.binary_cross_entropy_with_logits(pred[1], group_label, pos_weight=group_weights),
                    'loss/train_note': F.binary_cross_entropy_with_logits(pred[2], note_label),
                    'loss/train_tech': F.binary_cross_entropy_with_logits(pred[3], tech_label, pos_weight=tech_weights),
                    # 'loss/train_state': 3 * state_criterion(pred[0].reshape(-1,50), state_label.reshape(-1,50)),#, weight=state_weights),
                    'loss/train_LDS_l': lds_l,
                    'loss/train_LDS_ul': lds_ul,
                    'loss/train_r_norm_l': r_norm_l.abs().mean(),
                    'loss/train_r_norm_ul': r_norm_ul.abs().mean()                 
                    }
            return predictions, losses
        else:
            bins = state_label.shape[-2]   
            state_pred =  torch.sigmoid(pred[0])[:,:bins,:]
            group_pred = torch.sigmoid(pred[1])[:,:bins,:]
            note_pred = torch.sigmoid(pred[2])[:,:bins,:]
            tech_pred = torch.sigmoid(pred[3])[:,:bins,:]
            # state_pred = softmax(pred[0])
            # testing           
            state_pred = state_pred.reshape(-1, 3)
            state = []
            for s in state_pred:
                if s[s.argmax()] < 0.5:
                    state.append(0)
                else:
                    state.append(s.argmax())
            state_pred = torch.tensor(state)
            predictions = {
                    'note_state': state_pred.reshape(-1, bins),
                    'tech_group': group_pred.reshape(-1, 4).argmax(axis=1).reshape(-1, bins),
                    # 'note': note_pred.reshape(-1, bins),
                    'note': note_pred.reshape(-1, 50).argmax(axis=1).reshape(-1, bins),
                    'tech': tech_pred.reshape(-1, 10).argmax(axis=1).reshape(-1, bins),
                    'r_adv': r_adv,
                    }      
            losses = {
                    'loss/test_state': F.binary_cross_entropy_with_logits(pred[0][:,:bins,:], state_label),
                    'loss/test_group': F.binary_cross_entropy_with_logits(pred[1][:,:bins,:], group_label),
                    'loss/test_note': F.binary_cross_entropy_with_logits(pred[2][:,:bins,:], note_label),
                    'loss/test_tech': F.binary_cross_entropy_with_logits(pred[3][:,:bins,:], tech_label),
                    # 'loss/test_state': state_criterion(pred[0].reshape(-1,50), state_label.reshape(-1,50)),
                    'loss/test_LDS_l': lds_l,
                    'loss/test_r_norm_l': r_norm_l.abs().mean()                  
                    }
            return predictions, losses, (spec[:,1,:,:].squeeze(1), spec[:,3,:,:].squeeze(1)), pred, post, latent

    def load_my_state_dict(self, state_dict):
        """Useful when loading part of the weights. From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameterds
                param = param.data
            own_state[name].copy_(param)
