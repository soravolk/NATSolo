import os
from glob import glob
from tqdm import tqdm
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from abc import abstractmethod
from .constants import *

class PianoRollAudioDataset(Dataset):
    def __init__(self, path, folders=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        self.path = path
        self.folders = folders if folders is not None else self.available_folders()
        
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        self.data = []
        print(f"Loading {len(self.folders)} folder{'s' if len(self.folders) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for folder in self.folders:
            for input_files in tqdm(self.files(folder), desc='Loading folder %s' % folder): #self.files is defined in MAPS class
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
#             print(f'step_begin = {step_begin}')
            
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
#             print(f'begin = {begin}')
            end = begin + self.sequence_length
    
            result['audio'] = data['audio'][begin:end].to(self.device)
            if data.get('label') is not None:
              result['technique'] = data['label'][step_begin:step_end].to(self.device).float()
              result['start_idx'] = begin
            
        else:
            result['audio'] = data['audio'].to(self.device)
            result['technique'] = data['label'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15

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
        saved_data_path = audio_path.replace('.wav', '.pt')
        if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
        audio_length = len(audio)
        # unlabelled data
        if tsv_path is None:
          data = dict(path=audio_path, audio=audio)
          torch.save(data, saved_data_path)
          return data

        # !!!this may result in the unmatched time steps btn pred and label!!!
        n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps
        label = torch.zeros(n_steps, dtype=torch.uint8)

        # load labels(start, duration, techniques)
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        
        for start, end, technique in midi:
            left = int(round(start * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            left = min(n_steps, left) # Ensure the time step of onset would not exceed the last time step

            right = int(round(end * SAMPLE_RATE / HOP_LENGTH))
            right = min(n_steps, right) # Ensure the time step of frame would not exceed the last time step

            label[left:right] = technique

        data = dict(path=audio_path, audio=audio, label=label)
        torch.save(data, saved_data_path)
        # torch.save(data, saved_data_path)
        return data

class Solo(PianoRollAudioDataset):
    def __init__(self, path='./Solo', folders=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu'):
        self.overlap = overlap
        super().__init__(path, folders, sequence_length, seed, refresh, device)

    @classmethod
    def available_folders(cls):
        return ['train', 'test']
    
    def appending_wav_tsv(self, id_list, folder):
        wavs = list(glob(os.path.join(self.path, f"{folder}/wav", '*.wav')))
        wavs = sorted(wavs)
        if folder == 'train_unlabel':
          return wavs

        tsvs = list(glob(os.path.join(self.path, f"{folder}/tsv", '*.tsv')))
        tsvs = sorted(tsvs)   
        return wavs, tsvs

    def files(self, folder):
        if folder == 'train_label':
            types = np.arange(1, TOTAL_LABEL_FILES + 1)
            wavs, tsvs = self.appending_wav_tsv(types, folder)
        elif folder == 'train_unlabel':
            types = np.arange(1, TOTAL_UNLABEL_FILES + 1)
            wavs = self.appending_wav_tsv(types, folder)
            return zip(wavs)
        elif folder == 'valid':
            types = np.arange(1, TOTAL_VALID_FILES + 1)
            wavs, tsvs = self.appending_wav_tsv(types, folder)
        elif folder == 'test':
            types = np.arange(1, TOTAL_TEST_FILES + 1)
            wavs, tsvs = self.appending_wav_tsv(types, folder)

        assert(all(os.path.isfile(wav) for wav in wavs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return zip(wavs, tsvs)

def prepare_dataset(sequence_length, validation_length, refresh, device):
    # Choosing the dataset to use
    train_set = Solo(folders=['train_label'], sequence_length=sequence_length, device=device, refresh=refresh)
    valid_set = Solo(folders=['valid'], sequence_length=sequence_length, device=device, refresh=refresh)
    test_set = Solo(folders=['test'], sequence_length=sequence_length, device=device, refresh=refresh)

    return train_set, valid_set, test_set


def prepare_VAT_dataset(sequence_length, validation_length, refresh, device):
    l_set = Solo(folders=['train_label'], sequence_length=sequence_length, device=device)            
    ul_set = Solo(folders=['train_unlabel'], sequence_length=sequence_length, device=device) 
    valid_set = Solo(folders=['valid'], sequence_length=sequence_length, device=device)
    # what is full_validation??
    test_set = Solo(folders=['test'], sequence_length=None, device=device)
    
    return l_set, ul_set, valid_set, test_set