import os
import sys
from collections import defaultdict
from mir_eval.util import midi_to_hz
import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from scipy.stats import hmean
from tqdm import tqdm
from .convert import *
from .constants import *
from .utils import save_pianoroll

eps = sys.float_info.epsilon    

def evaluate_prediction(data, model, ep, save_path=None, reconstruction=True, tech_weights=None):
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

    metrics = defaultdict(list) # a safe dict

    song_count = 0
    for val_data in tqdm(data):
        pred, losses, _, _ = model.run_on_batch(val_data, None, False)
        
        # get label from one hot vector
        state_label = val_data['label'][:,:3].argmax(axis=1)
        group_label = val_data['label'][:,3:7].argmax(axis=1)
        note_label = val_data['label'][:,7:57].argmax(axis=1)
        tech_label = val_data['label'][:,57:].argmax(axis=1)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            if key in ['tech', 'note']:
                value.relu_() # value.shape [232, 10]

        ############ evaluate techniques ############
        # get the confusion matrix
        cm_dict = get_confusion_matrix(tech_label.cpu().numpy(), pred['tech'].squeeze(0).cpu().numpy())
        # get the recall and precision of techniques
        for key, value in technique_dict.items():
            p = cm_dict['Precision'][key - 1][key - 1]
            r = cm_dict['Recall'][key - 1][key - 1]
            f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
            metrics[f'metric/{value}/precision'].append(p)
            metrics[f'metric/{value}/recall'].append(r)
            metrics[f'metric/{value}/f1'].append(f)

        # find the technique timings
        # groundtruth: val_data['technique'].shape [232]

        pred['note'].squeeze_(0)
        pred['note_state'].squeeze_(0)
        # get techinique and interval
        tech_ref, tech_i_ref = extract_technique(tech_label, scale2time=True) # (tech_label, state_label)
        tech_est, tech_i_est = extract_technique(pred['tech'].squeeze(0), scale2time=True) # (pred['tech'], pred['note_state'])
        # get note and interval
        note_ref, note_i_ref = extract_notes(note_label, state_label, group_label, scale2time=True)
        note_est, note_i_est = extract_notes(pred['note'], pred['note_state'], pred['tech_group'].squeeze(0), scale2time=True)

        # save midi
        midi_path = os.path.join('midi', f'song{song_count}_ep{ep}.midi')
        save_midi(midi_path, note_est, note_i_est)
        # save ground truth midi
        gt_midi_path = os.path.join('midi', f'gt_song{song_count}.midi')
        save_midi(gt_midi_path, note_ref, note_i_ref)

        ############ evaluate notes ############
        # Converting midi number to frequency
        note_ref_hz = np.array([midi_to_hz(MIN_MIDI + midi) for midi in note_ref])
        note_est_hz = np.array([midi_to_hz(MIN_MIDI + midi) for midi in note_est])
        acc = evaluate_frame_accuracy(note_label, pred['note']) # frame level
        p, r, f, o = evaluate_notes(note_i_ref, note_ref_hz, note_i_est, note_est_hz, offset_ratio=None)
        metrics['metric/note/accuracy'].append(acc)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)
        # get note accuracy
        acc = evaluate_frame_accuracy_per_tech(tech_label, note_label, pred['note'])
        for key, value in technique_dict.items():
            a_single_tech = acc[key]
            metrics[f'metric/{value}/accuracy'].append(a_single_tech) # this is note_accuracy

        # p, r, f, o = evaluate_notes(note_i_ref, note_ref_hz, note_i_est, note_est_hz)
        # metrics['metric/note-with-offsets/precision'].append(p)
        # metrics['metric/note-with-offsets/recall'].append(r)
        # metrics['metric/note-with-offsets/f1'].append(f)
        # metrics['metric/note-with-offsets/overlap'].append(o)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.file.png')
            save_pianoroll(file_path, val_data['technique'])
            pred_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['technique'])
    
        song_count += 1
    return metrics

def eval_model(model, ep, loader, VAT_start=0, VAT=False, tech_weights=None):
    model.eval()
    batch_size = loader.batch_size
    metrics = defaultdict(list)
    i = 0 
    for batch in loader:
        if ep < VAT_start or VAT==False:
            predictions, losses, _, _ = model.run_on_batch(batch, None, False)
        else:
            predictions, losses, _, _ = model.run_on_batch(batch, None, True)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        print(f'Eval Epoch: {ep} [{i*batch_size}/{len(loader)*batch_size}'
                f'({100. * i / len(loader):.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}"
                , end='\r') 
        i += 1
    print(' '*100, end = '\r')          
    return metrics