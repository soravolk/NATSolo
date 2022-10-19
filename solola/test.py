import os
import sys
sys.path.append("..") 
from mir_eval.util import hz_to_midi, midi_to_hz
from collections import defaultdict
import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from scipy.stats import hmean
from model.convert import *
from model.dataset import Solo
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
def extract_note_tech(path):
    f = open(path, 'r')
    techniques = []
    t_intervals = []
    pitches = []
    p_intervals = []
    # get notes and result of every techniques
    for l in f.readlines():
        # print(l.strip('\n').split(' '))
        line = list(map(float, l.strip('\n').split(' ')))
        period = [line[1], line[1]+line[2]]
        pitches.append(int(line[0]))
        p_intervals.append(period)
        if int(line[3]) != 0 or int(line[4]) != 0 or int(line[5]) != 0:
            techniques.append(2) # bend
            t_intervals.append(period)
        if int(line[6]) != 0:
            techniques.append(5) # pull
            t_intervals.append(period)
        if int(line[7]) != 0:
            techniques.append(7) # hammer
            t_intervals.append(period)
        if int(line[8]) != 0 or int(line[9]) != 0 or int(line[10]) != 0:
            techniques.append(1) # slide
            t_intervals.append(period)
        if int(line[11]) != 0:
            techniques.append(3) # vibrato
            t_intervals.append(period)
    scaling = 160 / 16000
    org_pitches = np.array(pitches) - 40
    org_p_intervals = np.array([[round(s / scaling), round(e / scaling)] for (s, e) in p_intervals])
    org_t_intervals = np.array([[round(s / scaling), round(e / scaling)] for (s, e) in t_intervals])
    techniques = np.array(techniques) 
    t_intervals = np.array(t_intervals) 
    pitches = np.array([midi_to_hz(11 + midi) for midi in pitches])
    p_intervals = np.array(p_intervals)
    return techniques, t_intervals, pitches, p_intervals, org_pitches, org_p_intervals, org_t_intervals

def load_data():
    return Solo(path='../GN', folders=['valid'], sequence_length=None, device='cuda:0', audio_type='flac')

def techframe2pred(frame):
    pred = []
    for temp in frame:
        if len(temp) == 0:
            pred.append(0)
        else:
            pred.append(temp[0])
    return pred

def noteframe2pred(frame):
    pred = []
    for temp in frame:
        if len(temp) == 0:
            pred.append(0)
        else:
            pred.append(hz_to_midi(temp[0]) - 40)
    return pred

def evaluate(technique_dict, valid_set):
    metrics = defaultdict(list)
    result_dir = './old_licks'
    files = sorted(os.listdir(result_dir))
    macro_cm = None
    macro_note_label = []
    macro_state_label = []
    macro_note_pred = []
    macro_state_pred = []
    for file, valid in zip(files, valid_set):
        print(file, valid['path'])
        state_label = valid['label'][:,:2].argmax(axis=1)
        group_label = valid['label'][:,2:6].argmax(axis=1)
        note_label = valid['label'][:,6:56].argmax(axis=1)
        tech_label = valid['label'][:,56:].argmax(axis=1)

        tech_est, tech_i_est, note_est, note_i_est, org_note_est, org_note_i_est, org_tech_i_est = extract_note_tech(f'{result_dir}/{file}/FinalNotes.txt')
        note_ref, note_i_ref, org_note_ref, org_note_i_ref = extract_notes(note_label, state_label)
        # evaluate note-level
        p, r, f, o = evaluate_notes(note_i_ref, note_ref, note_i_est, note_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)
        # evaluate frame-level
        note_t_est, note_f_est, state_pred = notes_to_frames(org_note_est, org_note_i_est, note_label.shape, solola=True)
        note_t_ref, note_f_ref = notes_to_frames(org_note_ref, org_note_i_ref, note_label.shape)
        frame_metrics = evaluate_frames(note_t_ref, note_f_ref, note_t_est, note_f_est)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
        # evaluate technique
        # tech_ref, tech_i_ref = extract_technique(tech_label, scale2time=True)
        tech_t_ref, tech_f_ref = techniques_to_frames(tech_est, org_tech_i_est, tech_label.shape)
        tech_pred = techframe2pred(tech_f_ref)
        cm_dict = get_confusion_matrix(tech_label.cpu().numpy(), np.array(tech_pred), list(technique_dict.keys()))

        # get the recall and precision of techniques
        for key, value in technique_dict.items():
            p = cm_dict['Precision'][key][key]
            r = cm_dict['Recall'][key][key]
            f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
            metrics[f'metric/{value}/precision'].append(p)
            metrics[f'metric/{value}/recall'].append(r)
            metrics[f'metric/{value}/f1'].append(f)

        note_pred = noteframe2pred(note_f_ref)
        macro_note_label.extend(note_label)
        macro_state_label.extend(state_label)
        macro_note_pred.extend(note_pred)
        macro_state_pred.extend(state_pred)
        # sum up macro confusion matrix 
        if macro_cm is None:
            macro_cm = cm_dict['cm']
        else:
            macro_cm += cm_dict['cm']
            
    macro_note_label = torch.tensor(macro_note_label)
    macro_note_pred = torch.tensor(macro_note_pred)
    macro_note_ref, macro_note_i_ref, org_note_ref, org_note_i_ref = extract_notes(macro_note_label, torch.tensor(macro_state_label))
    macro_note_est, macro_note_i_est, org_note_est, org_note_i_est = extract_notes(macro_note_pred, torch.tensor(macro_state_pred))
    note_t_ref, note_f_ref = notes_to_frames(org_note_ref, org_note_i_ref, macro_note_label.shape)
    note_t_est, note_f_est = notes_to_frames(org_note_est, org_note_i_est, macro_note_pred.shape)
    
    acc = evaluate_frame_accuracy(macro_note_label, macro_note_pred) # frame level
    p, r, f, o = evaluate_notes(macro_note_i_ref, macro_note_ref, macro_note_i_est, macro_note_est, offset_ratio=None)
    metrics['metric/note/accuracy_macro'].append(acc)
    metrics['metric/note/precision_macro'].append(p)
    metrics['metric/note/recall_macro'].append(r)
    metrics['metric/note/f1_macro'].append(f)

    frame_metrics = evaluate_frames(note_t_ref, note_f_ref, note_t_est, note_f_est)
    metrics['metric/note/precision_frame_macro'].append(frame_metrics['Precision'])
    metrics['metric/note/recall_frame_macro'].append(frame_metrics['Recall'])
    metrics['metric/note/f1_frame_macro'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
    
    macro_recall, macro_precision  = get_prec_recall(macro_cm)
    for key, value in technique_dict.items():
        p = macro_precision[key][key]
        r = macro_recall[key][key]
        f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
        metrics[f'metric/{value}/macro_precision'].append(p)
        metrics[f'metric/{value}/macro_recall'].append(r)
        metrics[f'metric/{value}/macro_f1'].append(f)
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
        8: 'tap'
    }
    valid_set = load_data()
    metrics = evaluate(technique_dict, valid_set)
    with open(f'../testing/solola/metrics.txt', 'w') as f:
        for key, values in metrics.items():
            if key.startswith('metric/'):
                _, category, name = key.split('/')
                print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}', file=f)