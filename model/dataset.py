import os
import sys
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
from abc import abstractmethod
from .constants import *


def parse_tech_and_group(all_steps, all_tech, tech_label, tech_group_label, hop, sr):
    for start, end, technique in all_tech:
        # if technique == 0: # normal
        #     continue

        left = int(round(start * sr / hop)) # Convert time to time step
        left = min(all_steps, left) # Ensure the time step of onset would not exceed the last time step

        right = int((end * sr) // hop)
        right = min(all_steps, right) # Ensure the time step of frame would not exceed the last time step
        
        # silent_label[left - 2: right + 2] = 1 # not silent
        if technique in [1, 2, 3]: # slide, bend, trill
            tech_group_label[left:right] = 1
        elif technique in [5, 7, 8]: # pull, hammer, tap
            tech_group_label[left:right] = 2
        elif technique in [4, 6]: # harmonic, mute
            tech_group_label[left:right] = 3
        
        tech_label[left:right] = technique if technique != 0 else 9
    return tech_label, tech_group_label

def parse_note_and_state(all_steps, all_note, note_label, note_state_label, hop, sr):
    for start, end, note in all_note:
        left = int(round(start * sr / hop)) # Convert time to time step
        left = min(all_steps, left) # Ensure the time step of onset would not exceed the last time step

        right = int((end * sr) // hop)
        right = min(all_steps, right) # Ensure the time step of frame would not exceed the last time step

        if left + 3 > right:
            note_state_label[left: right - 1] = 1
            note_state_label[right - 1] = 0
        else:
            note_state_label[left: left + 4] = 1 # onset
            note_state_label[left + 4: right - 2] = 2 # activate
            note_state_label[right - 2: right] = 0

        note_label[left:right] = note
    return note_label - LOGIC_MIDI, note_state_label

def initialize_label(all_steps):
    tech_group_label = torch.zeros(all_steps, dtype=torch.int8)
    tech_label = torch.zeros(all_steps, dtype=torch.int8)
    note_state_label = torch.zeros(all_steps, dtype=torch.int8)
    note_label = torch.ones(all_steps, dtype=torch.int8) * 51
    return tech_group_label, tech_label, note_state_label, note_label

def gen_label(all_steps, tech_tsv_path, note_tsv_path, hop, sr):
    # 0 means silence (not lead guitar)
    tech_group_label, tech_label, note_state_label, note_label = initialize_label(all_steps)

    # load labels(start, duration, techniques)
    all_tech = np.loadtxt(tech_tsv_path, delimiter='\t', skiprows=1)
    all_note = np.loadtxt(note_tsv_path, delimiter='\t', skiprows=1)
    # processing tech labels
    tech_label, tech_group_label = parse_tech_and_group(all_steps, all_tech, tech_label, tech_group_label, hop, sr)
    # processing note labels
    note_label, note_state_label = parse_note_and_state(all_steps, all_note, note_label, note_state_label, hop, sr)
    
    ##### concat all one-hot label #####
    note_state_label_onehot = F.one_hot(note_state_label.to(torch.int64), num_classes=3)
    tech_group_label_onehot = F.one_hot(tech_group_label.to(torch.int64), num_classes=4)
    # 0 % 51 = 0 means no note (the lowest note is 52)
    note_label_onehot = F.one_hot(note_label.to(torch.int64), num_classes=50)
    tech_label_onehot = F.one_hot(tech_label.to(torch.int64), num_classes=10)
    label = torch.cat((note_state_label_onehot, tech_group_label_onehot, note_label_onehot, tech_label_onehot), 1)

    return note_state_label, tech_group_label, tech_label, label

class AudioDataset(Dataset):
    def __init__(self, path, folders=None, sequence_length=None, seed=42, refresh=False, device='cpu', audio_type='flac', sr=SAMPLE_RATE, hop=HOP_LENGTH):
        self.path = path
        self.folders = folders if folders is not None else self.available_folders()
        
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh
        self.audio_type = audio_type
        self.sr = sr
        self.hop = hop

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
            step_begin = self.random.randint(audio_length - self.sequence_length) // self.hop
            all_steps = self.sequence_length // self.hop
            step_end = step_begin + all_steps
            # trim audio time to fit the frame
            begin = step_begin * self.hop
            end = step_end * self.hop
            # end = begin + self.sequence_length
            result['audio'] = data['audio'][begin:end-1].to(self.device)
            # for labelled data
            if data.get('label') is not None:
              ####### why it should be float?? #######
              result['label'] = data['label'][step_begin:step_end].to(self.device).float()
            result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15 -> Dont know why
        else:
            result['audio'] = data['audio'].to(self.device).float().div_(32768.0)
            result['label'] = data['label'].to(self.device).float()
            result['tech_group_label'] = data['tech_group_label'].to(self.device).float()
            result['note_state_label'] = data['note_state_label'].to(self.device).float()
            result['tech_label'] = data['tech_label'].to(self.device).float()

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

    def load(self, audio_path, tech_tsv_path = None, note_tsv_path = None):
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
        if tech_tsv_path is None and note_tsv_path is None:
          data = dict(path=audio_path, audio=audio)
          torch.save(data, saved_data_path)
          return data

        # check if the annotation and audio are matched
        if self.audio_type == 'flac':
            assert(audio_path.split("/")[-1][:-4] == tech_tsv_path.split("/")[-1][:-3])
            assert(audio_path.split("/")[-1][:-4] == note_tsv_path.split("/")[-1][:-3])
        else:
            assert(audio_path.split("/")[-1][:-3] == tech_tsv_path.split("/")[-1][:-3])
            assert(audio_path.split("/")[-1][:-3] == note_tsv_path.split("/")[-1][:-3])

        # labels' time steps
        all_steps = int(audio_length // self.hop)
        # generate label
        note_state_label, tech_group_label, tech_label, label = gen_label(all_steps, tech_tsv_path, note_tsv_path, self.hop, self.sr)

        data = dict(path=audio_path, audio=audio, note_state_label=note_state_label, tech_group_label=tech_group_label, tech_label=tech_label, label=label)
        torch.save(data, saved_data_path)
        return data

class Solo(AudioDataset): #seed=42
    def __init__(self, path='./Solo', folders=None, sequence_length=None, overlap=True, seed=50, refresh=False, device='cpu', audio_type='flac', sr=SAMPLE_RATE, hop=HOP_LENGTH):
        super().__init__(path, folders, sequence_length, seed, refresh, device, audio_type, sr, hop)
        self.audio_type = audio_type

    @classmethod
    def available_folders(cls):
        return ['train_label', 'valid']
    
    def get_file_list(self, audio, audio_type):
        # make sure tsv and wav are matched
        tech_tsvs = [file.replace(f"{audio_type}/", "tech_tsv/").replace(f".{audio_type}", ".tsv") for file in audio]
        
        note_tsvs = [file.replace(f"{audio_type}/", "note_tsv/").replace(f".{audio_type}", ".tsv") for file in audio]    

        return tech_tsvs, note_tsvs

    def appending_wav_tsv(self, folder):
        audio = list(glob(os.path.join(self.path, f"{folder}/{self.audio_type}", f'*.{self.audio_type}')))
        audio = sorted(audio)

        tech_tsvs, note_tsvs = self.get_file_list(audio, self.audio_type)

        return audio, tech_tsvs, note_tsvs

    def files(self, folder):
        if folder == 'train_label':
            audio, tech_tsvs, note_tsvs = self.appending_wav_tsv(folder)
        elif folder == 'valid':
            audio, tech_tsvs, note_tsvs = self.appending_wav_tsv(folder)
        elif folder == 'test':
            audio, tech_tsvs, note_tsvs = self.appending_wav_tsv(folder)

        assert(all(os.path.isfile(wav) for wav in audio))
        assert(all(os.path.isfile(tsv) for tsv in tech_tsvs))
        assert(all(os.path.isfile(tsv) for tsv in note_tsvs))

        return zip(audio, tech_tsvs, note_tsvs)

def prepare_dataset(sequence_length, validation_length, refresh, device, audio_type, dataset):
    if dataset == 'Solo':
        l_set = Solo(folders=['train_label'], sequence_length=sequence_length, device=device, audio_type=audio_type)            
        valid_set = Solo(folders=['valid'], sequence_length=sequence_length, device=device, audio_type=audio_type)
        # full_validation (whole song)
        # test_set = Solo(folders=['test'], sequence_length=None, device=device, audio_type=audio_type)
        return l_set, valid_set
    else: 
        sys.exit("ERROR - Please provide a valid dataset")
        

def compute_dataset_weight(device):
    train_set = Solo(folders=['train_label'], sequence_length=None, device=device, refresh=None)

    y_1 = []
    y_2 = []
    y_3 = []
    for data in train_set:
        y_1.extend(data['tech_group_label'].detach().cpu().numpy())
        y_2.extend(data['note_state_label'].detach().cpu().numpy())
        y_3.extend(data['tech_label'].detach().cpu().numpy())
    tech_group_weights = compute_class_weight('balanced', np.unique(y_1), y_1)
    tech_group_weights = torch.tensor(tech_group_weights, dtype=torch.float).to(device)
    # tech_group_weights[2] = 4 * tech_group_weights[2]
    # note_state_weights = None
    note_state_weights = compute_class_weight('balanced', np.unique(y_2), y_2)
    note_state_weights = torch.tensor(note_state_weights, dtype=torch.float).to(device)
    note_state_weights[1] = 4 * note_state_weights[1] # weight more for onset
    note_state_weights[2] = 2 * note_state_weights[2]
    tech_weights = compute_class_weight('balanced', np.unique(y_3), y_3)
    tech_weights = torch.tensor(tech_weights, dtype=torch.float).to(device)

    return (note_state_weights, tech_group_weights, tech_weights)