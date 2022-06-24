import os
from glob import glob
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from torch import nn
from abc import abstractmethod
from .constants import *
from .feature_extraction import full_feature_extraction, cfp_feature_extraction
from .audio_augment import transform_method

class AudioDataset(Dataset):
    def __init__(self, path, folders=None, sequence_length=None, seed=42, refresh=False, device='cpu', audio_type='flac'):
        self.path = path
        self.folders = folders if folders is not None else self.available_folders()
        
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh
        self.audio_type = audio_type

        self.data = []
        print(f"Loading {len(self.folders)} folder{'s' if len(self.folders) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        print("Loading folder")
        for folder in self.folders:
            for input_files in tqdm(self.files(folder), desc='Loading folder %s' % folder):
                self.data.append(self.load(*input_files)) # self.load first loads all data into memory first
    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            # audio samples
            audio_length = len(data['audio'])
            # convert to time step
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            
            all_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + all_steps
            # trim audio time to fit the frame
            begin = step_begin * HOP_LENGTH
            end = step_end * HOP_LENGTH
            # end = begin + self.sequence_length
    
            result['audio'] = data['audio'][begin:end].to(self.device)
            # for labelled data
            if data.get('label') is not None:
              result['technique'] = data['label'][step_begin:step_end].to(self.device).float()

        else:
            result['audio'] = data['audio'].to(self.device)
            result['technique'] = data['label'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15 -> Dont know why

        return result

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_folders(cls):
        """return the names of all available folders"""
        raise NotImplementedError

    @abstractmethod
    def files(self, folder):
        """return the list of input files (audio_filename, tsv_filename) for this folder"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path = None):
        """
        load an audio track and the corresponding labels
        Returns
        -------
            A dictionary containing the following data:
            path: str
                the path to the audio file
            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform
            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace(f'.{self.audio_type}', '.pt')
        # if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
        #     return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE
        # convert numpy array to pytorch tensor
        # zero padding
        audio = torch.ShortTensor(audio) 
        if self.sequence_length is not None:
            diff = self.sequence_length - len(audio)
            if diff > 0:
                padding = nn.ConstantPad1d((0, diff + 1), 0)
                audio = padding(audio)
                print('padded: ', len(audio))
            
        audio_length = len(audio)

        # unlabelled data
        if tsv_path is None:
          data = dict(path=audio_path, audio=audio)
          torch.save(data, saved_data_path)
          return data

        # check if the annotation and audio are matched
        if self.audio_type == 'flac':
            assert(audio_path.split("/")[-1][:-4] == tsv_path.split("/")[-1][:-3])
        else:
            assert(audio_path.split("/")[-1][:-3] == tsv_path.split("/")[-1][:-3])


        # !!! This will affect the labels' time steps
        all_steps = audio_length  // HOP_LENGTH  
        # 0 means silence
        label = torch.zeros(all_steps, dtype=torch.uint8)

        # load labels(start, duration, techniques)
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        
        for start, end, technique in midi:
            left = int(round(start * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            left = min(all_steps, left) # Ensure the time step of onset would not exceed the last time step

            right = int(round(end * SAMPLE_RATE / HOP_LENGTH))
            right = min(all_steps, right) # Ensure the time step of frame would not exceed the last time step

            label[left:right] = technique
        data = dict(path=audio_path, audio=audio, label=label)
        torch.save(data, saved_data_path)
        return data

class Solo(AudioDataset):
    def __init__(self, path='./Solo', folders=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu', audio_type='wav'):
        super().__init__(path, folders, sequence_length, seed, refresh, device)
        self.audio_type = audio_type
    @classmethod
    def available_folders(cls):
        return ['train', 'test']
    
    def appending_wav_tsv(self, folder):
        wavs = list(glob(os.path.join(self.path, f"{folder}/{self.audio_type}", f'*.{self.audio_type}')))
        wavs = sorted(wavs)
        if folder == 'train_unlabel':
          return wavs

        # make sure tsv and wav are matched
        tsvs = []
        for file in wavs:
            if self.audio_type == 'flac':
                name = self.path + f"/{folder}/tsv/" + file.split("/")[-1][:-4] + 'tsv'
            else:
                name = self.path + f"/{folder}/tsv/" + file.split("/")[-1][:-3] + 'tsv'
            tsvs.append(name)

        return wavs, tsvs

    def files(self, folder):
        if folder == 'train_label':
            wavs, tsvs = self.appending_wav_tsv(folder)
        elif folder == 'train_unlabel':
            wavs = self.appending_wav_tsv(folder)
            return zip(wavs)
        elif folder == 'valid':
            wavs, tsvs = self.appending_wav_tsv(folder)
        elif folder == 'test':
            wavs, tsvs = self.appending_wav_tsv(folder)

        assert(all(os.path.isfile(wav) for wav in wavs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))
        
        return zip(wavs, tsvs)

class Solo_CFP(Solo):
    def __init__(self, path='./Solo', folders=None, sequence_length=None, sample_rate=16000, overlap=True,
                 seed=42, refresh=False, device='cpu', audio_type='wav', num_feat=9, k=9):
        self.overlap = overlap
        self.num_feat = num_feat
        super().__init__(path, folders, sequence_length, seed, refresh, device)
        self.normalize = lambda x: (x-np.mean(x))/(np.std(x)+1e-8)    
        self.sample_rate = sample_rate
        self.window_size = 2*k + 1  
        self.k = k 
    
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])
        audio = data['audio'][None, :, 0]
        feat = full_feature_extraction(*cfp_feature_extraction(audio, self.sample_rate))
        feat = feat.reshape(self.num_feat, 1566//self.num_feat, -1)
        feat = torch.Tensor(self.normalize(feat))

        result['audio'] = data['audio'].to(self.device)
        result['feature'] = feat.to(self.device)
        result['technique'] = data['label'].to(self.device).float()

        # result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15

        return result

class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(self, feature, num_feat=9, k=9):

        # --- Args ---
        self.window_size = 2*k+1
        self.k = k
        
        # --- Load File ---
        # self.feature = torch.from_numpy(feature).float()
        self.feature = feature
        self.feature = self.feature.reshape((num_feat,1566//num_feat,-1))
        self.len = self.feature.shape[-1]

        # --- Pad Length ---
        self.feature = torch.cat([
            torch.zeros((num_feat,1566//num_feat,k)),
            self.feature,
            torch.zeros((num_feat,1566//num_feat,k))
            ], dim=-1)
        
        # --- Transform ---
        self.transform_dict={'cutout'    :False,
                             'freq_mask' :{'freq_mask_param':100},
                             'time_mask' :False,
                             'pitchshift':{'shift_range':48}, 
                             'addnoise'  :False,
                             }
        self.data_aug = transform_method(self.transform_dict)
        self.data_normalize = lambda x: (x-torch.mean(x))/(torch.std(x)+1e-8)
        
        # --- Normalize ---
        self.feature = self.data_normalize(self.feature)
        # --- Augment ---
        self.feature = self.data_aug(self.feature.unsqueeze(0)).squeeze(0)
        
    def __getitem__(self, index):
        frame_feat = self.feature[:, :, index:index+self.window_size]
        return index, frame_feat
    
    def __len__(self):
        return self.len

def prepare_VAT_dataset(sequence_length, validation_length, refresh, device, audio_type):
    l_set = Solo(folders=['train_label'], sequence_length=sequence_length, device=device, audio_type=audio_type)            
    ul_set = Solo(folders=['train_unlabel'], sequence_length=sequence_length, device=device, audio_type=audio_type) 
    valid_set = Solo(folders=['valid'], sequence_length=sequence_length, device=device, audio_type=audio_type)
    # what is full_validation??
    test_set = Solo(folders=['test'], sequence_length=None, device=device, audio_type=audio_type)
    
    return l_set, ul_set, valid_set, test_set

def prepare_CFP_dataset(sequence_length, validation_length, refresh, device, audio_type):
    l_set = Solo_CFP(folders=['train_label'], sequence_length=sequence_length, sample_rate=SAMPLE_RATE, device=device, audio_type=audio_type)            
    ul_set = Solo_CFP(folders=['train_unlabel'], sequence_length=sequence_length, sample_rate=SAMPLE_RATE, device=device, audio_type=audio_type) 
    valid_set = Solo_CFP(folders=['valid'], sequence_length=sequence_length, sample_rate=SAMPLE_RATE, device=device, audio_type=audio_type)
    # what is full_validation??
    test_set = Solo_CFP(folders=['test'], sequence_length=None, sample_rate=SAMPLE_RATE, device=device, audio_type=audio_type)
    return l_set, ul_set, valid_set, test_set

def prepare_dataset(sequence_length, validation_length, refresh, device):
    # Choosing the dataset to use
    train_set = Solo(folders=['train_label'], sequence_length=sequence_length, device=device, refresh=refresh)
    valid_set = Solo(folders=['valid'], sequence_length=sequence_length, device=device, refresh=refresh)
    test_set = Solo(folders=['test'], sequence_length=sequence_length, device=device, refresh=refresh)

def compute_dataset_weight(device):
    train_set = Solo_CFP(folders=['train_label'], sequence_length=None, device=device, refresh=None)

    y = []
    for data in train_set:
        print('data.technique.shape', data['technique'].shape)
        y.extend(data['technique'].detach().cpu().numpy())
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    return class_weights