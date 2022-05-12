import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import scipy.ndimage
import math
import matplotlib.pyplot as plt

def transform_method(transform_dict):
    
    # Cut-out, Frequency Masking, Pitch shift ... etc
    
    transform_list = []
    
    try:
        if transform_dict['pitchshift'] != False:
            transform_list.append(PitchShifting(**transform_dict['pitchshift']))
        if transform_dict['addnoise'] != False:
            transform_list.append(AddNoise(**transform_dict['addnoise']))
        if transform_dict['cutout'] != False:
            transform_list.append(CutOut(**transform_dict['cutout']))
        if transform_dict['freq_mask'] != False:
            transform_list.append(FrequencyMasking(**transform_dict['freq_mask']))
        if transform_dict['time_mask'] != False:
            transform_list.append(TimeMasking(**transform_dict['time_mask']))
    except:
        raise ValueError("""
            transform_method() should contain a full dictionary with: 
            transform_dict={\'cutout\'    :{\'n_holes\':[cutout holes], \'height\':[cutout height], \'width\':[cutout width]}, 
                            \'freq_mask\' :{\'freq_mask_param\':[F parameter in SpecAugment]},
                            \'time_mask\' :{\'time_mask_param\':[T parameter in SpecAugment]},
                            \'pitchshift\':{\'shift_range\':2},
                             }
            for transforms unused, simply give a bool \'False\' for the dictionary key
            """)
    
    return transforms.Compose(transform_list)

def image_show(torch_Tensor):
    torch_Tensor = torch_Tensor.numpy()
    torch_Tensor = torch_Tensor.reshape((174*9,19))
    plt.matshow(np.repeat(torch_Tensor, 100, 1))
    return

class CutOut(object):
    '''
    Better with normalized input
    Switched to rectangular due to the input shape of (522,19)
    '''
    def __init__(self, n_holes, height, width):
        self.n_holes = n_holes
        self.height  = height
        self.width   = width
    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        
        mask = np.ones((h,w), np.float32)
        
        for holes in range(self.n_holes):
            centre_y = np.random.randint(h)
            centre_x = np.random.randint(w)
            
            y1 = np.clip(centre_y - self.height // 2, 0, h)
            y2 = np.clip(centre_y + self.height // 2, 0, h)
            x1 = np.clip(centre_x - self.width  // 2, 0, w)
            x2 = np.clip(centre_x + self.width  // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img  = img * mask.type_as(img)
        
        return img
    
class FrequencyMasking(object):
    '''
    Better with normalized input
    '''
    def __init__(self, freq_mask_param):
        self.F = freq_mask_param
    def __call__(self, img):
        v    = img.size(2)
        f    = np.random.randint(0, self.F)
        f0   = np.random.randint(0, v-f)
        img[:,:,f0:f0+f,:].fill_(0.)        
        return img

class TimeMasking(object):
    '''
    Better with normalized input
    '''
    def __init__(self, time_mask_param):
        self.T = time_mask_param
    def __call__(self, img):
        tau  = img.size(3)
        t    = np.random.randint(0, self.T)
        t0   = np.random.randint(0, tau-t)
        img[:,:,:,t0:t0+t].fill_(0.)        
        return img

class PitchShifting(object):
    def __init__(self, shift_range):
        self.shift_range = shift_range
    def __call__(self, img):
        f_range   = img.size(2)
        shift     = np.random.randint(-self.shift_range, self.shift_range+1)
        shift_img = torch.zeros(img.size()).type_as(img)
        for f in range(f_range):
            if f-shift >= 0 and f-shift < f_range:
                shift_img[:,:, f] = img[:,:, f-shift]
        shift_img = shift_img.float()
        return shift_img
    
class AddNoise(object):
    def __init__(self, noise_type, noise_size):
        self.noise_type = noise_type
        self.noise_size = torch.FloatTensor(1).uniform_(0,noise_size)[0]
    def __call__(self, img):
        '''
        
        to make mean=0, variance=1, 
        we need uniform(-sqrt(3), sqrt(3))
        
        '''
        if self.noise_type == 'pink':
            f_range = img.size(2)
            gen_noise = np.empty(img.size())
            
            for f in range(f_range):
                gen_noise[:,:,f] = np.random.uniform(-np.sqrt(3)*f/f_range, np.sqrt(3)*f/f_range, size=img.size(3))
            gen_noise = torch.from_numpy(gen_noise).float().type_as(img)
            return img + self.noise_size * gen_noise
        elif self.noise_type == 'white':
            gen_noise = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=img.size())
            gen_noise = torch.from_numpy(gen_noise).float().type_as(img)
            return img + self.noise_size*gen_noise

class EasyClipping(object):
    '''
    Simply clip to sign(x) randomly
    '''
    def __init__(self, prob, threshold):
        self.prob = prob
        self.threshold = threshold
        
    def __call__(self, img):
        random_num = torch.Tensor([0]).uniform_(0,1)
        if random_num < self.prob:
            img = torch.clamp(img, -self.threshold, self.threshold)           
        return img
 
"""
class Clipping(object):
    '''
    Not yet correct...
    '''
    def __init__(self, threshold, clip_ratio, hard_clip = False):
        self.clip_ratio = clip_ratio # not sure
        self.hard_clip  = hard_clip
        self.threshold  = threshold
                
        # really ugly, but means the harmonic indices of each frequency band index
        self.idx_dict = {i : [np.clip(np.round(np.log(2*n+3)/np.log(12.5)*174 + i).astype(int),0,173) for n in range( (math.floor(12.5**(1-i/174))-3) //2 + 1 ) if 12.5 ** (i/174 -1)*(2*n+3) < 1] for i in range(174)}
        
    def __call__(self, img):
        f_range   = img.size(1) // 3
        clip_img = torch.zeros(img.size())
        clip_pos = torch.gt(img, self.threshold)
        for r in range(img.size(0)):
            for t in range(img.size(2)):
                for feat in range(3):
                    for f in range(f_range*feat, f_range*(feat+1)):
                        if clip_pos[r,f,t] == 1:
                            clip_img[r,f,t] += self.threshold - img[r,f,t]
                            for harmonics in range(len(self.idx_dict(f%f_range))):
                                clip_img[r,f+self.idx_dict[harmonics],t] += 1./harmonics * img[r,f,t]                    
        clip_img = clip_img.float()
        cmix_img = img + clip_img
        cmix_img = (cmix_img-torch.mean(cmix_img))/torch.std(cmix_img)
        return cmix_img
"""