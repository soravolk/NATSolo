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

def evaluate_pitch_frame_and_note_level(note_label, state_label, note_pred, state_pred, tech_label, metrics, technique_dict, scaling, macro=False, poly=False):
    m = '_macro' if macro else ''
    note_ref, note_i_ref, org_note_ref, org_note_i_ref = extract_notes(note_label, state_label, scaling=scaling)
    note_est, note_i_est, org_note_est, org_note_i_est = extract_notes(note_pred, state_pred, scaling=scaling, poly=poly)

    note_t_ref, note_f_ref = notes_to_frames(org_note_ref, org_note_i_ref, note_label.shape, scaling)
    note_t_est, note_f_est = notes_to_frames(org_note_est, org_note_i_est, note_pred.shape, scaling)
    ############ evaluate notes ############
    # frame level acc
    if not poly:
        acc = evaluate_frame_accuracy(note_label, note_pred) 
        metrics[f'metric/note/accuracy{m}'].append(acc)
    # note level p, r, f1
    p, r, f, o = evaluate_notes(note_i_ref, note_ref, note_i_est, note_est, strict=False, offset_ratio=None)
    metrics[f'metric/note/precision{m}'].append(p)
    metrics[f'metric/note/recall{m}'].append(r)
    metrics[f'metric/note/f1{m}'].append(f)
    # frame level p, r, f1
    frame_metrics = evaluate_frames(note_t_ref, note_f_ref, note_t_est, note_f_est)
    metrics[f'metric/note/precision_frame{m}'].append(frame_metrics['Precision'])
    metrics[f'metric/note/recall_frame{m}'].append(frame_metrics['Recall'])
    metrics[f'metric/note/f1_frame{m}'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
    # get note accuracy
    if tech_label is not None and not poly:
        acc = evaluate_frame_accuracy_per_tech(tech_label, note_label, note_pred)
        for key, value in technique_dict.items():
            a_single_tech = acc[key]
            metrics[f'metric/{value}/accuracy{m}'].append(a_single_tech) # this is note_accuracy

    # p, r, f, o = evaluate_notes(note_i_ref, note_ref_hz, note_i_est, note_est_hz)
    # metrics['metric/note-with-offsets/precision'].append(p)
    # metrics['metric/note-with-offsets/recall'].append(r)
    # metrics['metric/note-with-offsets/f1'].append(f)
    # metrics['metric/note-with-offsets/overlap'].append(o)
    return metrics

def calculate_tech_recall_precision(cm_dict, technique_dict, metrics, macro):
    m = 'macro_' if macro else ''
    for key, value in technique_dict.items():
        p = cm_dict['Precision'][key][key]
        r = cm_dict['Recall'][key][key]
        f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
        metrics[f'metric/{value}/{m}precision'].append(p)
        metrics[f'metric/{value}/{m}recall'].append(r)
        metrics[f'metric/{value}/{m}f1'].append(f)
    return metrics

def evaluate_technique_frame_and_note_level(label, pred, metrics, macro_cm, technique_dict, scaling, macro=False, solola=False):
    m = '_macro' if macro else ''
    if macro:
        macro_recall, macro_precision  = get_prec_recall(macro_cm)
        cm_dict = {
            'Precision': macro_precision,
            'Recall': macro_recall
        }
        cm_dict_all = {}
        cm_dict_all['cm'] = macro_cm
        cm_dict_all['Precision'] = macro_precision
        cm_dict_all['Recall'] = macro_recall
    else:
        cm_dict = get_confusion_matrix(label.cpu().numpy(), pred.cpu().numpy(), list(technique_dict.keys()))
        # sum up macro confusion matrix
        if macro_cm is None:
            macro_cm = cm_dict['cm']
        else:
            macro_cm += cm_dict['cm']
    # get the recall and precision of techniques
    metrics = calculate_tech_recall_precision(cm_dict, technique_dict, metrics, macro=macro)

    # get techinique and interval
    tech_ref, tech_i_ref, org_tech_i_ref = extract_technique(label, states=None, scale2time=True, scaling=scaling) # (tech_label, state_label)
    tech_est, tech_i_est, org_tech_i_est = extract_technique(pred, states=None, scale2time=True, scaling=scaling) # (pred['tech'], pred['note_state'])

    metrics = evaluate_technique_note_level(tech_ref, tech_i_ref, tech_est, tech_i_est, technique_dict, metrics, macro=macro)
    
    tech_t_ref, tech_f_ref = techniques_to_frames(tech_ref, org_tech_i_ref, label.shape, scaling)
    tech_t_est, tech_f_est = techniques_to_frames(tech_est, org_tech_i_est, pred.shape, scaling)
    # print(len(tech_t_ref), len(tech_f_ref), len(tech_t_est), len(tech_f_ref))
    # frame level p, r, f1
    frame_metrics = evaluate_frames(tech_t_ref, tech_f_ref, tech_t_est, tech_f_est)
    metrics[f'metric/technique/precision_frame{m}'].append(frame_metrics['Precision'])
    metrics[f'metric/technique/recall_frame{m}'].append(frame_metrics['Recall'])
    metrics[f'metric/technique/f1_frame{m}'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    if macro:
        return metrics, cm_dict_all
    else:
        return metrics, macro_cm

def evaluate_prediction(data, model, ep, technique_dict, scaling, save_path=None, reconstruction=False, testing=False, has_state=True, has_group=True, eval_note=True, eval_tech=True):
    metrics = defaultdict(list) # a safe dict
    macro_cm = None
    cm_dict_all = None
    macro_note_label = [] if eval_note else None
    macro_note_pred = []  if eval_note else None
    macro_state_label = [] if has_state else None
    macro_state_pred = [] if has_state else None
    macro_tech_label = [] if eval_tech else None
    macro_tech_pred = [] if eval_tech else None
    val_loss = None
        
    if testing:
        inference = defaultdict(list)
        specs = []
        probs = []

    for val_data in tqdm(data):
        total_loss = 0
        pred, losses, spec, prob, _, _ = model.run_on_batch(val_data, None, False)
        # if val_loss is None:
        #     val_loss = defaultdict(list)
        #     for key, value in {**losses}.items():
        #         val_loss[key] = value
        # else:
        #     for key, value in {**losses}.items():
        #         val_loss[key] += value             
        # get label from one hot vector
        ######## for state of size 3 ########
        state_label = val_data['label'][:,:3].argmax(axis=1)
        group_label = val_data['label'][:,3:7].argmax(axis=1)
        note_label = val_data['label'][:,7:57].argmax(axis=1)
        tech_label = val_data['label'][:,57:].argmax(axis=1)

        for key, loss in losses.items():
            metrics[key].append(loss.item())
        for key, value in pred.items():
            if key in ['tech', 'note']:
                value.relu_() # value.shape [232, 10]

        note_pred = pred['note'].squeeze(0) if eval_note else None
        state_pred = pred['note_state'].squeeze(0) if has_state else None
        
        tech_pred = pred['tech'].squeeze(0) if eval_tech else None
        ############ evaluate techniques ############
        # get the confusion matrix
        if eval_tech:
            metrics, macro_cm = evaluate_technique_frame_and_note_level(tech_label, tech_pred, metrics, macro_cm, technique_dict, scaling)

        if eval_note:
            macro_note_label.extend(note_label)
            macro_note_pred.extend(note_pred)
        if has_state:
            macro_state_label.extend(state_label)
            macro_state_pred.extend(state_pred)
        if eval_tech:
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
                specs.append(spec[0].squeeze(0).cpu().numpy())
                probs.append((prob[4].squeeze(0).cpu().numpy(), prob[2].squeeze(0).cpu().numpy())) # note_prob, note_prob passed to attention with state

        ############ evaluate notes ############ 
        if eval_note:
            metrics = evaluate_pitch_frame_and_note_level(note_label, state_label, note_pred, state_pred, tech_label, metrics, technique_dict, scaling)

    ############ get the macro recall and precision of techniques ############ 
    if eval_tech:
        macro_tech_label = torch.tensor(macro_tech_label)
        macro_tech_pred = torch.tensor(macro_tech_pred)
        metrics, cm_dict_all = evaluate_technique_frame_and_note_level(macro_tech_label, macro_tech_pred, metrics, macro_cm, technique_dict, scaling, macro=True)
 
    ############ get macro note metrics ############
    if eval_note:
        if has_state:
            macro_state_label = torch.tensor(macro_state_label)
            macro_state_pred = torch.tensor(macro_state_pred)
        macro_note_label = torch.tensor(macro_note_label)
        macro_note_pred = torch.tensor(macro_note_pred)
        metrics = evaluate_pitch_frame_and_note_level(macro_note_label, macro_state_label, macro_note_pred, macro_state_pred, macro_tech_label, metrics, technique_dict, scaling, macro=True)

    if not testing:
        return metrics, cm_dict_all #, val_loss
    else:
        return metrics, cm_dict_all, inference, specs, probs

def eval_model(model, ep, loader, VAT_start=0, VAT=False):
    model.eval()
    batch_size = loader.batch_size
    metrics = defaultdict(list)
    i = 0 
    for batch in loader:
        if ep < VAT_start or VAT==False:
            predictions, losses, _, _, _, _ = model.run_on_batch(batch, None, False)
        else:
            predictions, losses, _, _, _, _ = model.run_on_batch(batch, None, True)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        print(f'Eval Epoch: {ep} [{i*batch_size}/{len(loader)*batch_size}'
                f'({100. * i / len(loader):.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}"
                , end='\r') 
        i += 1
    print(' '*100, end = '\r')          
    return metrics