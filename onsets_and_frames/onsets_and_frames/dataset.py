import os
from glob import glob
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
from abc import abstractmethod
from .constants import *

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
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            # trim audio time to fit the frame
            begin = step_begin * HOP_LENGTH
            end = step_end * HOP_LENGTH
            # begin = step_begin * HOP_LENGTH
            # end = begin + self.sequence_length

            # result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)

            result['audio'] = data['audio'][begin:end-1].to(self.device)
            #   result['label'] = data['label'][step_begin:step_end].to(self.device).float()
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        
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

    def load(self, audio_path, note_tsv_path = None):
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
        # saved_data_path = audio_path.replace(f'.{self.audio_type}', '.pt')
        # if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
        #     return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE
        # convert numpy array to pytorch tensor
        # zero padding
        audio = torch.ShortTensor(audio) 
        audio_length = len(audio)
        if self.sequence_length is not None:
            diff = self.sequence_length - len(audio)
            if diff > 0:
                padding = nn.ConstantPad1d((0, diff + 1), 0)
                audio = padding(audio)
                print('padded: ', len(audio))

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        # check if the annotation and audio are matched
        if self.audio_type == 'flac':
            assert(audio_path.split("/")[-1][:-4] == note_tsv_path.split("/")[-1][:-3])
        else:
            assert(audio_path.split("/")[-1][:-3] == note_tsv_path.split("/")[-1][:-3])

        # labels' time steps
        all_steps = audio_length // HOP_LENGTH
        # 0 means silence (not lead guitar)
        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        # load labels(start, duration, techniques)
        all_note = np.loadtxt(note_tsv_path, delimiter='\t', skiprows=1)

        # processing note labels
        for onset, offset, note in all_note:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - LOGIC_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            # left = int(round(start * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            # left = min(all_steps, left) # Ensure the time step of onset would not exceed the last time step
            # right = int((end * SAMPLE_RATE) // HOP_LENGTH)
            # right = min(all_steps, right) # Ensure the time step of frame would not exceed the last time step
            # if left + 3 > right:
            #     note_state_label[left: right - 1] = 1
            #     # note_state_label[right - 1] = 0
            # else:
            #     note_state_label[left: left + 4] = 1 # onset
            #     # note_state_label[left + 4: right - 2] = 2 # activate
            #     # note_state_label[right - 2: right] = 0

            # if left - 1 > 0:
            #     note_state_label[left-1] = 1
            # if left + 3 > right:
            #     note_state_label[left: right] = note
            # else:
            #     note_state_label[left: left + 4] = note # onset

            # note_label[left:right] = note
        
        ##### concat all one-hot label #####
        # note_state_label_onehot = F.one_hot(note_state_label.to(torch.int64), num_classes=2)
        # 0 % 51 = 0 means no note (the lowest note is 52)
        # note_label_onehot = F.one_hot(note_label.to(torch.int64) - 51, num_classes=50)
        # label = torch.cat((note_state_label_onehot, note_label_onehot), 1)
        # label = torch.cat((note_state_label.unsqueeze(1), tech_group_label.unsqueeze(1), (note_label - 51).unsqueeze(1), tech_label.unsqueeze(1)), 1)
        data = dict(path=audio_path, audio=audio, label=label)
        # torch.save(data, saved_data_path)
        return data

class Solo(AudioDataset):
    def __init__(self, path='../Solo', folders=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu', audio_type='wav'):
        super().__init__(path, folders, sequence_length, seed, refresh, device)
        self.audio_type = audio_type
    @classmethod
    def available_folders(cls):
        return ['train', 'test']
    
    def appending_wav_tsv(self, folder):
        audio = list(glob(os.path.join(self.path, f"{folder}/{self.audio_type}", f'*.{self.audio_type}')))
        audio = sorted(audio)
        if folder == 'train_unlabel':
          return audio

        # make sure tsv and wav are matched
        note_tsvs = []
        for file in audio:
            if self.audio_type == 'flac':
                note_name = self.path + f"/{folder}/note_tsv/" + file.split("/")[-1][:-4] + 'tsv'
            else:
                note_name = self.path + f"/{folder}/note_tsv/" + file.split("/")[-1][:-4] + 'tsv'
            note_tsvs.append(note_name)

        return audio, note_tsvs

    def files(self, folder):
        if folder == 'train_label':
            audio, note_tsvs = self.appending_wav_tsv('train_label')
        elif folder == 'valid':
            audio, note_tsvs = self.appending_wav_tsv('valid')
        # elif folder == 'valid':

        assert(all(os.path.isfile(wav) for wav in audio))
        assert(all(os.path.isfile(tsv) for tsv in note_tsvs))

        return zip(audio, note_tsvs)
