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

def plot_predicted_transcription(specs, probs, note_interval, note, tech_interval, tech, state, save_folder):
    tech_trans = {
        0: '0',
        1: 's',
        2: 'b',
        3: '~',
        4: 'x',
        5: 'p',
        6: '<>',
        7: 'h',
        8: 't',
        9: 'n'
    }
    # specs = specs.cpu().detach().numpy()
    for i, (spec, prob, x, y, x_tech, y_tech, onset) in enumerate(zip(specs, probs, note_interval, note, tech_interval, tech, state)):
        time_range = spec.shape[0] * (HOP_LENGTH/SAMPLE_RATE)
        fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(100,20), gridspec_kw={'height_ratios': [4,1]})
        ax = ax.flat
        # spectrogram
        # librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        # ax[0].set_ylabel('f (Hz)', fontsize=70)
        # ax[0].tick_params(labelsize=65)
        # label_format = '{:,.0f}'
        # ticks_loc = ax[0].get_yticks().tolist()
        # ax[0].set_ylim([512, 4096])
        # ax[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # ax[0].set_yticklabels([label_format.format(x) for x in ticks_loc])
        # # note prob
        # ax[1].imshow(np.flip(prob[0].transpose(), 0), cmap='plasma')
        # ax[1].set_xlim([0, spec.shape[0]])
        # ax[1].axis('off')
        # # note prob refined by state
        # ax[2].imshow(np.flip(prob[1].transpose(), 0), cmap='plasma')
        # ax[2].set_xlim([0, spec.shape[0]])
        # ax[2].axis('off')
        # note transcription
        ax[0].tick_params(labelbottom=False, labelsize=65)
        ax[0].set_ylabel('midi #', fontsize=70)
        ax[0].set_xlim([0, time_range])
        ax[0].set_ylim([60, 100])
        # ax[3].set_yticklabels([70, 80, 90])
        label_format = '{:,.0f}'
        ticks_loc = ax[0].get_yticks().tolist()
        ax[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax[0].set_yticklabels([label_format.format(x) for x in ticks_loc])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[0].plot(x_val, y_val, linewidth=7.5)
            # ax[1].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # onset information
        for o in onset:
            ax[0].vlines(o, ymin=60, ymax=100, linestyles='dotted')
        # techique transcription
        # ax[4].set_xlabel('time (s)', fontsize=37)
        # ax[4].set_ylabel('technique', fontsize=37)
        ax[1].tick_params(labelleft=False, labelsize=60)#, labelbottom=False)
        ax[1].set_xlim([0, time_range])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[1].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=45)
            ax[1].plot(x_val, y_val, linewidth=7.5)
            ax[1].vlines(t[0], ymin=0.7, ymax=2, linestyles='dotted')
        plt.savefig(f'{save_folder}/prediction/{i}.png')
        plt.close()

def plot_groundtruth_transcription(specs, note_interval, note, tech_interval, tech, state, save_folder):
    tech_trans = {
        0: '0',
        1: 's',
        2: 'b',
        3: '~',
        4: 'x',
        5: 'p',
        6: '<>',
        7: 'h',
        8: 't',
        9: 'n'
    }
    # specs = specs.cpu().detach().numpy()
    for i, (spec, x, y, x_tech, y_tech, onset) in enumerate(zip(specs, note_interval, note, tech_interval, tech, state)):
        time_range = spec.shape[0] * (HOP_LENGTH/SAMPLE_RATE)
        fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(100,20), gridspec_kw={'height_ratios': [4, 1]})
        ax = ax.flat
        # spectrogram
        # librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        # ax[0].set_ylabel('spectrogram')
        # note transcription
        ax[0].tick_params(labelbottom=False, labelsize=65)
        ax[0].set_ylabel('midi # (GT)', fontsize=70)
        ax[0].set_xlim([0, time_range])
        ax[0].set_ylim([60, 100])
        label_format = '{:,.0f}'
        ticks_loc = ax[0].get_yticks().tolist()
        ax[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax[0].set_yticklabels([label_format.format(x) for x in ticks_loc])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[0].plot(x_val, y_val, linewidth=7.5)
            # ax[0].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # onset information
        for o in onset:
            ax[0].vlines(o, ymin=60, ymax=100, linestyles='dotted')
        # techique transcription
        ax[1].tick_params(labelleft=False, labelsize=60)
        ax[1].set_xlabel('time (s)', fontsize=70)
        # ax[1].set_ylabel('technique', fontsize=60)
        ax[1].set_xlim([0, time_range])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[1].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=45)
            ax[1].plot(x_val, y_val, linewidth=7.5)
            ax[1].vlines(t[0], ymin=0.7, ymax=2, linestyles='dotted')
        plt.savefig(f'{save_folder}/groundtruth/{i}.png')
        plt.close()
        
def save_transcription_and_midi(inferences, spec, prob, save_folder, scaling):
    transcription_path = f'{save_folder}/transcription'
    midi_path = f'{save_folder}/midi'

    transcriptions = defaultdict(list)
    note_labels = inferences['testing_note_label']
    state_labels = inferences['testing_state_label']
    tech_labels = inferences['testing_tech_label']
    note_preds = inferences['testing_note_pred']
    state_preds = inferences['testing_state_pred']
    tech_preds = inferences['testing_tech_pred']
    ep = 1
    for i, (s_label, s_pred, n_label, n_pred, t_label, t_pred) in enumerate(zip(state_labels, state_preds, note_labels, note_preds, tech_labels, tech_preds)):
        transcriptions = gen_transcriptions_and_midi(transcriptions, s_label, s_pred, n_label, n_pred, t_label, t_pred, i, ep, midi_path, scaling=scaling)

    plot_groundtruth_transcription(spec, transcriptions['note_interval_gt'], transcriptions['note_gt'], transcriptions['tech_interval_gt'], transcriptions['tech_gt'], transcriptions['state_gt'], transcription_path)
    plot_predicted_transcription(spec, prob, transcriptions['note_interval'], transcriptions['note'], transcriptions['tech_interval'], transcriptions['tech'], transcriptions['state'], transcription_path)

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
    testing = True
    inference = defaultdict(list)
    specs = []
    probs = []
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
        
        if testing:
            inference['testing_note_label'].append(note_label)
            inference['testing_state_label'].append(state_label)
            inference['testing_note_pred'].append(note_pred)
            inference['testing_state_pred'].append(state_pred)
            if eval_note and eval_tech:
                inference['testing_tech_label'].append(tech_label)
                inference['testing_tech_pred'].append(tech_pred)
                # specs.append(spec[0].squeeze(0).cpu().numpy())
                # probs.append((prob[4].squeeze(0).cpu().numpy(), prob[2].squeeze(0).cpu().numpy())) # note_prob, note_prob passed to attention with state

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

    return metrics, inference, specs, probs, scaling

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
    metrics, inferences, specs, probs, scaling = evaluate(technique_dict, valid_set, args.model)
    save_transcription_and_midi(inferences, specs, probs, args.save_path, scaling)
    # with open(f'{args.save_path}/metrics.txt', 'w') as f:
    #     for key, values in metrics.items():
    #         if key.startswith('metric/'):
    #             _, category, name = key.split('/')
    #             print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}', file=f)
    #             print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')