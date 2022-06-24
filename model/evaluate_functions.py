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
        1: 'normal', 
        2: 'slide',
        3: 'bend',
        4: 'trill',
        5: 'mute',
        6: 'pull',
        7: 'harmonic',
        8: 'hammer',
        9: 'tap'
    }

    metrics = defaultdict(list) # a safe dict
    transcriptions = defaultdict(list)

    for val_data in tqdm(data):
        pred, losses, _ = model.run_on_batch(val_data, None, False, tech_weights)
        tech_label = val_data['label'][:,:10].argmax(axis=1) # only one label file
        note_label = val_data['label'][:,10:].argmax(axis=1)
        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            if key in ['tech', 'note']:
                value.relu_() # value.shape [232, 10]

        ############ evaluate techniques ############
        # get the confusion matrix
        correct_tech_labels = tech_label.cpu().numpy()
        predict_tech_labels = pred['tech'].squeeze(0).cpu().numpy()
        
        cm, cm_recall, cm_precision = get_confusion_matrix(correct_tech_labels, predict_tech_labels)
        # get the recall and precision of techniques
        for key, value in technique_dict.items():
            p = cm_precision[key][key]
            r = cm_recall[key][key]
            f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
            metrics[f'metric/{value}/precision'].append(p)
            metrics[f'metric/{value}/recall'].append(r)
            metrics[f'metric/{value}/f1'].append(f)

        # find the technique timings
        # groundtruth: val_data['technique'].shape [232]
        scaling = HOP_LENGTH / SAMPLE_RATE

        # get techinique and interval
        pred['tech'].squeeze_(0)
        tech_ref, tech_i_ref = extract_technique(tech_label)
        tech_est, tech_i_est = extract_technique(pred['tech'])
        tech_i_ref = (tech_i_ref * scaling).reshape(-1, 2)
        tech_i_est = (tech_i_est * scaling).reshape(-1, 2)
        ############ evaluate notes ############
        pred['note'].squeeze_(0)
        note_ref, note_i_ref = extract_notes(note_label)
        note_est, note_i_est = extract_notes(pred['note'])

        # Converting time steps to seconds and midi number to frequency

        note_i_ref = (note_i_ref * scaling).reshape(-1, 2)
        note_ref_hz = np.array([midi_to_hz(MIN_MIDI + midi) for midi in note_ref])
        note_i_est = (note_i_est * scaling).reshape(-1, 2)
        note_est_hz = np.array([midi_to_hz(MIN_MIDI + midi) for midi in note_est])
       
        p, r, f, o = evaluate_notes(note_i_ref, note_ref_hz, note_i_est, note_est_hz, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)     

        p, r, f, o = evaluate_notes(note_i_ref, note_ref_hz, note_i_est, note_est_hz)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        # may implement frame_metrics later

        cm_dict = {
            'cm': cm,
            'Precision': cm_precision,
            'Recall': cm_recall,
        }
        if ep == 1:
            transcriptions['tech'].append(tech_ref)
            transcriptions['tech_interval'].append(tech_i_ref)
            transcriptions['note'].append(note_ref + 52)
            transcriptions['note_interval'].append(note_i_ref)
        else:
            transcriptions['tech'].append(tech_est)
            transcriptions['tech_interval'].append(tech_i_est)
            transcriptions['note'].append(note_est + 52)
            transcriptions['note_interval'].append(note_i_est)

        # if reconstruction:
        #     correct_tech_labels = val_data['technique'].cpu().numpy()
        #     predict_tech_labels = pred['technique2'].squeeze(0).cpu().numpy()
        #     cm_2, cm_recall_2, cm_precision_2 = get_confusion_matrix(correct_tech_labels, predict_tech_labels)
        #     # get the recall and precision of techniques
        #     for key, value in technique_dict.items():
        #         p = cm_precision_2[key][key]
        #         r = cm_recall_2[key][key]
        #         f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
        #         metrics[f'metric/technique/{value}/precision_2'].append(p)
        #         metrics[f'metric/technique/{value}/recall_2'].append(r)
        #         metrics[f'metric/technique/{value}/f1_2'].append(f)

        #     cm_dict = {
        #         'cm': cm,
        #         'Precision': cm_precision,
        #         'Recall': cm_recall,
        #         'cm_2': cm_2,
        #         'Precision_2': cm_precision_2,
        #         'Recall_2': cm_recall_2
        #     }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.file.png')
            save_pianoroll(file_path, val_data['technique'])
            pred_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['technique'])
    return metrics, cm_dict, transcriptions

def eval_model(model, ep, loader, VAT_start=0, VAT=False, tech_weights=None):
    model.eval()
    batch_size = loader.batch_size
    metrics = defaultdict(list)
    i = 0 
    for batch in loader:
        if ep < VAT_start or VAT==False:
            predictions, losses, _ = model.run_on_batch(batch, None, False, tech_weights)
        else:
            predictions, losses, _ = model.run_on_batch(batch, None, True, tech_weights)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        print(f'Eval Epoch: {ep} [{i*batch_size}/{len(loader)*batch_size}'
                f'({100. * i / len(loader):.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}"
                , end='\r') 
        i += 1
    print(' '*100, end = '\r')          
    return metrics