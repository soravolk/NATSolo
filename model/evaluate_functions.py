import os
import sys
from collections import defaultdict

import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_techniques
from sklearn.metrics import average_precision_score
from scipy.stats import hmean
from tqdm import tqdm
from .convert import *
from .constants import *
from .utils import save_pianoroll

eps = sys.float_info.epsilon    

def evaluate_wo_velocity(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None, reconstruction=True, onset=True, pseudo_onset=False, rule='rule2', VAT=False):
    metrics = defaultdict(list)
    
    for file in tqdm(data):
        print('file.shape: ', file['audio'].shape)
        pred, losses, _ = model.run_on_batch(file, None, False)   
#         print(f"pred['onset2'] = {pred['onset2'].shape}")
#         print(f"pred['frame2'] = {pred['frame2'].shape}")            

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            if key in ['technique', 'technique2']:
                value.squeeze_(0).relu_()
        # find the technique timings
        file['technique'] = file['technique'].unsqueeze(1)
        print("file['technique'].shape: ", file['technique'].shape)
        # get techinique and interval
        p_ref, i_ref = extract_technique(file['technique'], rule=rule, gt=True)
        p_est, i_est = extract_technique(pred['technique'], rule=rule)

        scaling = HOP_LENGTH / SAMPLE_RATE

        # Converting time steps to seconds and midi number to frequency
        i_ref = (i_ref * scaling).reshape(-1, 2)
        print("i_ref.shape: ", i_ref.shape)
        i_est = (i_est * scaling).reshape(-1, 2)
        print("i_est.shape: ", i_est.shape)

        p, r, f, o = evaluate_techniques(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        print('p: ', p)
        print('r: ', r)
        print('f: ', f)
        metrics['metric/technique/precision'].append(p)
        metrics['metric/technique/recall'].append(r)
        metrics['metric/technique/f1'].append(f)
        metrics['metric/technique/overlap'].append(o)     

        p, r, f, o = evaluate_techniques(i_ref, p_ref, i_est, p_est)
        metrics['metric/technique-with-offsets/precision'].append(p)
        metrics['metric/technique-with-offsets/recall'].append(r)
        metrics['metric/technique-with-offsets/f1'].append(f)
        metrics['metric/technique-with-offsets/overlap'].append(o)
     
        # frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        # metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        ## !!!multiclass format is not supported!!!
        # avp = average_precision_score(file['technique'].cpu().detach().flatten() ,pred['technique'].cpu().detach().flatten())
        # metrics['metric/MusicNet/micro_avg_P'].append(avp)     
        
        if reconstruction:
            p_est2, i_est2 = extract_technique(pred['technique2'], rule=rule)   
            # t_est2, f_est2 = techniques_to_frames(p_est2, i_est2, pred['frame2'].shape)               

            i_est2 = (i_est2 * scaling).reshape(-1, 2)
            # p_est2 = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est2])        

            # t_est2 = t_est2.astype(np.float64) * scaling
            # f_est2 = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est2]

            p2, r2, f2, o2 = evaluate_techniques(i_ref, p_ref, i_est2, p_est2, offset_ratio=None)
            metrics['metric/technique/precision_2'].append(p2)
            metrics['metric/technique/recall_2'].append(r2)
            metrics['metric/technique/f1_2'].append(f2)
            metrics['metric/technique/overlap_2'].append(o2)             

            # frame_metrics2 = evaluate_frames(t_ref, f_ref, t_est2, f_est2)
            # frame_metrics['Precision_2'] = frame_metrics2['Precision']
            # frame_metrics['Recall_2'] = frame_metrics2['Recall']
            # frame_metrics['accuracy_2'] = frame_metrics2['Accuracy']
            # metrics['metric/frame/f1_2'].append(hmean([frame_metrics['Precision_2'] + eps, frame_metrics['Recall_2'] + eps]) - eps)            
            # avp = average_precision_score(file['technique'].cpu().detach().flatten() ,pred['frame2'].cpu().detach().flatten())
            # metrics['metric/MusicNet/micro_avg_P2'].append(avp)   
            
            p2, r2, f2, o2 = evaluate_techniques(i_ref, p_ref, i_est2, p_est2)
            metrics['metric/technique-with-offsets/precision_2'].append(p2)
            metrics['metric/technique-with-offsets/recall_2'].append(r2)
            metrics['metric/technique-with-offsets/f1_2'].append(f2)
            metrics['metric/technique-with-offsets/overlap_2'].append(o2)             
            
        # for key, loss in frame_metrics.items():
        #     metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, os.path.basename(file['path']) + '.file.png')
            save_pianoroll(file_path, file['technique'])
            pred_path = os.path.join(save_path, os.path.basename(file['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['technique'])
    return metrics

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