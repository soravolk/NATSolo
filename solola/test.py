import os
import sys
import argparse
sys.path.append("..") 
from mir_eval.util import hz_to_midi, midi_to_hz
from collections import defaultdict
import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from scipy.stats import hmean
from model.convert import *
from model.constants import *
from model.evaluate_functions import *
from model.dataset import Solo, parse_tech_and_group, parse_note_and_state
'''
0: pitch(midi number)
1: onset
2: duration
3, 4, 5: bend
6: pull-off
7: hammer-on
8, 9, 10: slide
11: vibrato
'''

eps = sys.float_info.epsilon

def parse():
    parser = argparse.ArgumentParser(description='solola testing script')
    parser.add_argument('model', type=str, help='The result to which model to evaluate')
    parser.add_argument('save_path', type=str, default='testing', help='Path to save midi, transcriptions and corresponding metrics')
    return parser

def parse_poly_note_and_state(all_steps, all_note, note_label, note_state_label, hop, sr):
    for start, end, note in all_note:
        left = int(round(start * sr / hop)) # Convert time to time step
        left = min(all_steps, left) # Ensure the time step of onset would not exceed the last time step

        right = int((end * sr) // hop)
        right = min(all_steps, right) # Ensure the time step of frame would not exceed the last time step

        note_idx = int(note - LOGIC_MIDI)
        if left + 3 > right:
            note_state_label[left:right - 1, note_idx] = 1
            note_state_label[right - 1, note_idx] = 0
        else:
            note_state_label[left:left + 4, note_idx] = 1 # onset
            note_state_label[left +4:right - 2, note_idx] = 2 # activate
            note_state_label[right - 2:right, note_idx] = 0
    
        note_label[left:right, note_idx] = 1
    return note_label, note_state_label

def extract_note_tech(note_tech_pred):
    f = open(note_tech_pred, 'r')
    techniques = []
    pitches = [] # midi number
    
    for l in f.readlines():
        # print(l.strip('\n').split(' '))
        normal_flag = True
        line = list(map(float, l.strip('\n').split(' ')))
        # [begin, end, label/pred]
        pitches.append([line[1], line[1]+line[2], int(line[0]) + 12])
        if int(line[3]) != 0 or int(line[4]) != 0 or int(line[5]) != 0:
            normal_flag = False
            techniques.append([line[1], line[1]+line[2], 2]) # bend
        if int(line[6]) != 0:
            normal_flag = False
            if len(techniques) != 0 and techniques[-1][0] == line[1]:
                temp_end = techniques[-1][1] 
                half = techniques[-1][0] + (temp_end - techniques[-1][0]) / 2
                techniques[-1][1] = half
                techniques.append([half, temp_end, 5])
            else:
                techniques.append([line[1], line[1]+line[2], 5]) # pull
        if int(line[7]) != 0:
            normal_flag = False
            if len(techniques) != 0 and techniques[-1][0] == line[1]:
                temp_end = techniques[-1][1] 
                half = techniques[-1][0] + (temp_end - techniques[-1][0]) / 2
                techniques[-1][1] = half
                techniques.append([half, temp_end, 7])
            else:
                techniques.append([line[1], line[1]+line[2], 7]) # hammer
        if int(line[8]) != 0 or int(line[9]) != 0 or int(line[10]) != 0:
            normal_flag = False
            if len(techniques) != 0 and techniques[-1][0] == line[1]:
                temp_end = techniques[-1][1] 
                half = techniques[-1][0] + (temp_end - techniques[-1][0]) / 2
                techniques[-1][1] = half
                techniques.append([half, temp_end, 1])
            else:
                techniques.append([line[1], line[1]+line[2], 1]) # slide
        if int(line[11]) != 0:
            normal_flag = False
            if len(techniques) != 0 and techniques[-1][0] == line[1]:
                temp_end = techniques[-1][1] 
                half = techniques[-1][0] + (temp_end - techniques[-1][0]) / 2
                techniques[-1][1] = half
                techniques.append([half, temp_end, 3])
            else:
                techniques.append([line[1], line[1]+line[2], 3]) # vibrato
        if normal_flag:
            techniques.append([line[1], line[1]+line[2], 9]) # normal
    # techniques, t_intervals = process_tech(techniques, t_intervals)
    techniques = np.array(techniques)
    pitches = np.array(pitches)
    return techniques, pitches


def load_data(model):
    if model == 'solola':
        # return Solo(path='../GN', folders=['valid'], sequence_length=None, device='cuda:0', audio_type='wav', sr=SOLOLA_SAMPLE_RATE, hop=SOLOLA_HOP_LENGTH)
        return Solo(path='../Solo', folders=['valid'], sequence_length=None, device='cuda:0', audio_type='wav', sr=SOLOLA_SAMPLE_RATE, hop=SOLOLA_HOP_LENGTH)
    elif model == 'bp':
        return Solo(path='../GN', folders=['valid'], sequence_length=None, device='cuda:0', audio_type='wav', sr=BP_SAMPLE_RATE, hop=BP_HOP_LENGTH)
    elif model == 'mt3':
        return Solo(path='../GN', folders=['valid'], sequence_length=None, device='cuda:0', audio_type='wav', sr=16000, hop=128)

def evaluate(technique_dict, valid_set, model):
    metrics = defaultdict(list)
    macro_cm = None
    macro_note_label = []
    macro_state_label = []
    # macro_note_pred = []
    # macro_state_pred = []
    macro_note_pred = torch.tensor([])
    macro_state_pred = torch.tensor([])
    macro_tech_label = []
    macro_tech_pred = []
    if model == 'solola':
        sr = SOLOLA_SAMPLE_RATE
        hop = SOLOLA_HOP_LENGTH
        eval_note = True
        eval_tech = True
        result_dir = './new_testing'
        # result_dir = './old_licks'
        poly = False
    elif model == 'bp':
        sr = BP_SAMPLE_RATE
        hop= BP_HOP_LENGTH
        eval_note = True
        eval_tech = False
        result_dir = './bp_note_pred'
        poly = True
    elif model == 'mt3':
        sr = 16000
        hop= 128
        eval_note = True
        eval_tech = False
        result_dir = './mt3_note_pred'
        poly = True
    scaling = hop / sr
    files = sorted(os.listdir(result_dir))

    for valid in valid_set:
        file = valid['path'].split('/')[-1][:-4]
        if model == 'solola':
            eval_path = f'{result_dir}/{file}/FinalNotes.txt'
        elif model in ['bp', 'mt3']:
            eval_path = f'{result_dir}/{file}.tsv'

        if not os.path.isfile(eval_path):
            continue
        print(file, valid['path'])

        if model == 'solola':
            all_tech, all_note = extract_note_tech(eval_path)
        elif model in ['bp', 'mt3']:
            all_note = np.loadtxt(eval_path, delimiter='\t', skiprows=1)

        audio_length = len(valid['audio'])
        all_steps = int(audio_length // hop)
        if eval_note:
            tech_label = None
            state_label = valid['label'][:,:3].argmax(axis=1)
            note_label = valid['label'][:,7:57].argmax(axis=1)
            # processing note preds
            if model == 'solola': # monophonic
                state_pred = torch.zeros(all_steps, dtype=torch.int8) # dummy, will not use
                note_pred = torch.ones(all_steps, dtype=torch.int8) * 51
                note_pred, state_pred = parse_note_and_state(all_steps, all_note, note_pred, state_pred, hop, sr)
            else:
                state_pred = torch.zeros(all_steps, 50, dtype=torch.int8) # dummy, will not use
                note_pred = torch.zeros(all_steps, 50, dtype=torch.int8)
                note_pred, state_pred = parse_poly_note_and_state(all_steps, all_note, note_pred, state_pred, hop, sr)
            # evaluate pitch
            metrics = evaluate_pitch_frame_and_note_level(note_label, state_label, note_pred, state_pred, tech_label, metrics, technique_dict, scaling, poly=poly)
            macro_note_label.extend(note_label)
            macro_state_label.extend(state_label)
            macro_note_pred = torch.cat((macro_note_pred, note_pred), 0)
            macro_state_pred = torch.cat((macro_state_pred, state_pred), 0)
            # macro_note_pred.extend(note_pred)
            # macro_state_pred.extend(state_pred)
        if eval_tech:
            group_label = valid['label'][:,3:7].argmax(axis=1)
            tech_label = valid['label'][:,57:].argmax(axis=1)
            tech_group_pred = torch.zeros(all_steps, dtype=torch.int8) # dummy, will not use
            tech_pred = torch.zeros(all_steps, dtype=torch.int8)
            # processing tech preds
            tech_pred, tech_group_pred = parse_tech_and_group(all_steps, all_tech, tech_pred, tech_group_pred, hop, sr)
            # evaluate technique
            metrics, macro_cm = evaluate_technique_frame_and_note_level(tech_label, tech_pred, metrics, macro_cm, technique_dict, scaling)
            macro_tech_label.extend(tech_label)
            macro_tech_pred.extend(tech_pred)

    if eval_note:
        macro_note_label = torch.tensor(macro_note_label)
        # macro_note_pred = torch.tensor(macro_note_pred)
        macro_state_label = torch.tensor(macro_state_label)
        # macro_state_pred = torch.tensor(macro_state_pred)
        metrics = evaluate_pitch_frame_and_note_level(macro_note_label, macro_state_label, macro_note_pred, macro_state_pred, macro_tech_label, metrics, technique_dict, scaling, macro=True, poly=poly)
    if eval_tech:
        macro_tech_label = torch.tensor(macro_tech_label)
        macro_tech_pred = torch.tensor(macro_tech_pred)
        metrics, cm_dict_all = evaluate_technique_frame_and_note_level(macro_tech_label, macro_tech_pred, metrics, macro_cm, technique_dict, scaling, macro=True)

    return metrics

if __name__ == '__main__':
    technique_dict = {
        0: 'no tech',
        1: 'slide',
        2: 'bend',
        3: 'trill',
        4: 'mute',
        5: 'pull',
        6: 'harmonic',
        7: 'hammer',
        8: 'tap',
        9: 'normal'
    }
    parser = parse()
    args = parser.parse_args()
    valid_set = load_data(args.model)
    metrics = evaluate(technique_dict, valid_set, args.model)
    with open(f'{args.save_path}/metrics.txt', 'w') as f:
        for key, values in metrics.items():
            if key.startswith('metric/'):
                _, category, name = key.split('/')
                print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}', file=f)
                print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')