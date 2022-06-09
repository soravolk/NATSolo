import os
import sys
from collections import defaultdict

import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_techniques
from scipy.stats import hmean
from tqdm import tqdm
from .convert import *
from .constants import *
from .utils import save_pianoroll

eps = sys.float_info.epsilon    

def evaluate_prediction(data, model, save_path=None, reconstruction=True):
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

    for val_data in tqdm(data):
        pred, losses, _ = model.run_on_batch(val_data, None, False)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            if key in ['technique', 'technique2']:
                value.relu_() # value.shape [232, 10]

        # find the technique timings
        # groundtruth: val_data['technique'].shape [232]

        # get techinique and interval
        # tech_ref, interval_ref = extract_technique(val_data['technique'], gt=True)
        # tech_est, interval_est = extract_technique(pred['technique'].squeeze(0))
        
        # get the confusion matrix
        correct_labels = val_data['technique'].cpu().numpy()
        predict_labels = pred['technique'].squeeze(0).cpu().numpy()
        
        cm, cm_recall, cm_precision = get_confusion_matrix(correct_labels, predict_labels)
        # get the recall and precision of techniques
        for key, value in technique_dict.items():
            p = cm_precision[key][key]
            r = cm_recall[key][key]
            f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
            metrics[f'metric/technique/{value}/precision'].append(p)
            metrics[f'metric/technique/{value}/recall'].append(r)
            metrics[f'metric/technique/{value}/f1'].append(f)

        # scaling = HOP_LENGTH / SAMPLE_RATE

        # Converting time steps to seconds and midi number to frequency
        # interval_ref = (interval_ref * scaling) #[# of techniques, 2(onset, offset)]
        # interval_est = (interval_est * scaling)

        # utilize mir_eval(have to shift all technique by 1 because 0 is not allowed)
        # p, r, f, o = evaluate_techniques(interval_ref, tech_ref, interval_est, tech_est, offset_ratio=None)
        # metrics['metric/technique/precision'].append(p)
        # metrics['metric/technique/recall'].append(r)
        # metrics['metric/technique/f1'].append(f)
        # metrics['metric/technique/overlap'].append(o)     

        # p, r, f, o = evaluate_techniques(interval_ref, tech_ref, interval_est, tech_est)
        # metrics['metric/technique-with-offsets/precision'].append(p)
        # metrics['metric/technique-with-offsets/recall'].append(r)
        # metrics['metric/technique-with-offsets/f1'].append(f)
        # metrics['metric/technique-with-offsets/overlap'].append(o)
        
        if reconstruction:
            correct_labels = val_data['technique'].cpu().numpy()
            predict_labels = pred['technique2'].squeeze(0).cpu().numpy()
            cm_2, cm_recall_2, cm_precision_2 = get_confusion_matrix(correct_labels, predict_labels)
            # get the recall and precision of techniques
            for key, value in technique_dict.items():
                p = cm_precision_2[key][key]
                r = cm_recall_2[key][key]
                f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
                metrics[f'metric/technique/{value}/precision_2'].append(p)
                metrics[f'metric/technique/{value}/recall_2'].append(r)
                metrics[f'metric/technique/{value}/f1_2'].append(f)

            cm_dict = {
                'cm': cm,
                'Precision': cm_precision,
                'Recall': cm_recall,
                'cm_2': cm_2,
                'Precision_2': cm_precision_2,
                'Recall_2': cm_recall_2
            }

            # tech_est2, interval_est2 = extract_technique(pred['technique2'].squeeze(0))                
            # interval_est2 = (interval_est2 * scaling).reshape(-1, 2)  

            # p2, r2, f2, o2 = evaluate_techniques(interval_ref, tech_ref, interval_est2, tech_est2, offset_ratio=None)
            # metrics['metric/technique/precision_2'].append(p2)
            # metrics['metric/technique/recall_2'].append(r2)
            # metrics['metric/technique/f1_2'].append(f2)
            # metrics['metric/technique/overlap_2'].append(o2)             
            
            # p2, r2, f2, o2 = evaluate_techniques(interval_ref, tech_ref, interval_est2, tech_est2)
            # metrics['metric/technique-with-offsets/precision_2'].append(p2)
            # metrics['metric/technique-with-offsets/recall_2'].append(r2)
            # metrics['metric/technique-with-offsets/f1_2'].append(f2)
            # metrics['metric/technique-with-offsets/overlap_2'].append(o2)
        else:
            cm_dict = {
                'cm': cm,
                'Precision': cm_precision,
                'Recall': cm_recall,
            }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.file.png')
            save_pianoroll(file_path, val_data['technique'])
            pred_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['technique'])
    return metrics, cm_dict

def eval_model(model, ep, loader, VAT_start=0, VAT=False):
    model.eval()
    batch_size = loader.batch_size
    metrics = defaultdict(list)
    i = 0 
    for batch in loader:
        if ep < VAT_start or VAT==False:
            predictions, losses, _ = model.run_on_batch(batch, None, False)
        else:
            predictions, losses, _ = model.run_on_batch(batch, None, True)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        print(f'Eval Epoch: {ep} [{i*batch_size}/{len(loader)*batch_size}'
                f'({100. * i / len(loader):.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}"
                , end='\r') 
        i += 1
    print(' '*100, end = '\r')          
    return metrics