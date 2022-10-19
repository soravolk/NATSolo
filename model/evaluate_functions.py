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

def evaluate_prediction(data, model, ep, technique_dict, save_path=None, reconstruction=True, testing=False, eval_note=True, eval_tech=True):
    metrics = defaultdict(list) # a safe dict
    macro_cm = None
    macro_note_label = []
    macro_state_label = []
    macro_tech_label = []
    macro_note_pred = []
    macro_state_pred = []
    macro_tech_pred = []
    val_loss = None
    if len(data) > 5:
        iteration = 1
    else:
        iteration = 1
        
    if testing:
        inference = defaultdict(list)
        specs = []
        probs = []

    for val_data in tqdm(data):
        total_loss = 0
        temp_metrics = defaultdict(list)
        for i in range(iteration):
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
            ######## for state of size 2 ########
            # state_label = val_data['label'][:,:2].argmax(axis=1)
            # group_label = val_data['label'][:,2:6].argmax(axis=1)
            # note_label = val_data['label'][:,6:56].argmax(axis=1)
            # tech_label = val_data['label'][:,56:].argmax(axis=1)
            ######## for state of size 50 ########
            # state_label = val_data['label'][:,:50].argmax(axis=1)
            # group_label = val_data['label'][:,50:54].argmax(axis=1)
            # note_label = val_data['label'][:,54:104].argmax(axis=1)
            # tech_label = val_data['label'][:,104:].argmax(axis=1)
            ######## for softmax ########
            # val_data['label'] = val_data['label'].type(torch.LongTensor).cuda()
            # state_label = val_data['label'][:,0]#.argmax(axis=1)
            # group_label = val_data['label'][:,1]#.argmax(axis=1)
            # note_label = val_data['label'][:,2]#.argmax(axis=1)
            # tech_label = val_data['label'][:,3]#.argmax(axis=1)
            for key, loss in losses.items():
                temp_metrics[key].append(loss.item())
            for key, value in pred.items():
                if key in ['tech', 'note']:
                    value.relu_() # value.shape [232, 10]

            pred['note'].squeeze_(0)
            pred['note_state'].squeeze_(0)
            pred['tech'].squeeze_(0)
            ############ evaluate techniques ############
            # get the confusion matrix
            if eval_tech:
                cm_dict = get_confusion_matrix(tech_label.cpu().numpy(), pred['tech'].cpu().numpy(), list(technique_dict.keys()))
            if eval_tech:
                # sum up macro confusion matrix
                if i == 0:
                    if macro_cm is None:
                        macro_cm = cm_dict['cm']
                    else:
                        macro_cm += cm_dict['cm']
                # get the recall and precision of techniques
                for key, value in technique_dict.items():
                    p = cm_dict['Precision'][key][key]
                    r = cm_dict['Recall'][key][key]
                    f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
                    temp_metrics[f'metric/{value}/precision'].append(p)
                    temp_metrics[f'metric/{value}/recall'].append(r)
                    temp_metrics[f'metric/{value}/f1'].append(f)
                # get techinique and interval
                tech_ref, tech_i_ref = extract_technique(tech_label, scale2time=True) # (tech_label, state_label)
                tech_est, tech_i_est = extract_technique(pred['tech'], scale2time=True) # (pred['tech'], pred['note_state'])
                
                temp_metrics = evaluate_technique(tech_ref, tech_i_ref, tech_est, tech_i_est, technique_dict, temp_metrics)

            if i == 0:
                macro_note_label.extend(note_label)
                macro_state_label.extend(state_label)
                macro_note_pred.extend(pred['note'])
                macro_state_pred.extend(pred['note_state'])
                macro_tech_label.extend(tech_label)
                macro_tech_pred.extend(pred['tech'])
                if testing:
                    inference['testing_note_label'].append(note_label)
                    inference['testing_state_label'].append(state_label)
                    inference['testing_note_pred'].append(pred['note'])
                    inference['testing_state_pred'].append(pred['note_state'])
                    if eval_note and eval_tech:
                        inference['testing_tech_label'].append(tech_label)
                        inference['testing_tech_pred'].append(pred['tech'])
                        specs.append(spec[0].squeeze(0).cpu().numpy())
                        probs.append((prob[4].squeeze(0).cpu().numpy(), prob[2].squeeze(0).cpu().numpy())) # note_prob, note_prob passed to attention with state
            ############ evaluate techniques ############
            # tp, tr, tf, to = evaluate_notes(tech_i_ref, tech_ref, tech_i_est, tech_est, strict=False, offset_ratio=None, pitch_tolerance=0)
            # temp_metrics['metric/tech/precision'].append(tp)
            # temp_metrics['metric/tech/recall'].append(tr)
            # temp_metrics['metric/tech/f1'].append(tf)
            # temp_metrics['metric/tech/overlap'].append(to)

            ############ get note and interval ############ 
            if eval_note:
                note_ref, note_i_ref, org_note_ref, org_note_i_ref = extract_notes(note_label, state_label)
                note_est, note_i_est, org_note_est, org_note_i_est = extract_notes(pred['note'], pred['note_state'])
                note_t_ref, note_f_ref = notes_to_frames(org_note_ref, org_note_i_ref, note_label.shape)
                note_t_est, note_f_est = notes_to_frames(org_note_est, org_note_i_est, pred['note'].shape)
                ############ evaluate notes ############
                # frame level acc
                acc = evaluate_frame_accuracy(note_label, pred['note']) 
                # note level p, r, f1
                p, r, f, o = evaluate_notes(note_i_ref, note_ref, note_i_est, note_est, strict=False, offset_ratio=None)
                temp_metrics['metric/note/accuracy'].append(acc)
                temp_metrics['metric/note/precision'].append(p)
                temp_metrics['metric/note/recall'].append(r)
                temp_metrics['metric/note/f1'].append(f)
                temp_metrics['metric/note/overlap'].append(o)
                # frame level p, r, f1
                frame_metrics = evaluate_frames(note_t_ref, note_f_ref, note_t_est, note_f_est)
                temp_metrics['metric/note/precision_frame'].append(frame_metrics['Precision'])
                temp_metrics['metric/note/recall_frame'].append(frame_metrics['Recall'])
                temp_metrics['metric/note/f1_frame'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
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
        for key, loss in losses.items():
            metrics[key].append(sum(temp_metrics[key]) / iteration)
        if eval_tech:
            for key, value in technique_dict.items():
                metrics[f'metric/{value}/precision'].append(sum(temp_metrics[f'metric/{value}/precision']) / iteration)
                metrics[f'metric/{value}/recall'].append(sum(temp_metrics[f'metric/{value}/recall']) / iteration)
                metrics[f'metric/{value}/f1'].append(sum(temp_metrics[f'metric/{value}/f1']) / iteration)
                metrics[f'metric/{value}/precision_note'].append(sum(temp_metrics[f'metric/{value}/precision_note']) / iteration)
                metrics[f'metric/{value}/recall_note'].append(sum(temp_metrics[f'metric/{value}/recall_note']) / iteration)
                metrics[f'metric/{value}/f1_note'].append(sum(temp_metrics[f'metric/{value}/f1_note']) / iteration)
        if eval_note:
            metrics['metric/note/accuracy'].append(sum(temp_metrics['metric/note/accuracy']) / iteration)
            metrics['metric/note/precision'].append(sum(temp_metrics['metric/note/precision']) / iteration)
            metrics['metric/note/recall'].append(sum(temp_metrics['metric/note/recall']) / iteration)
            metrics['metric/note/f1'].append(sum(temp_metrics['metric/note/f1']) / iteration)
            metrics['metric/note/overlap'].append(sum(temp_metrics['metric/note/overlap']) / iteration)
            metrics['metric/note/precision_frame'].append(sum(temp_metrics['metric/note/precision_frame']) / iteration)
            metrics['metric/note/recall_frame'].append(sum(temp_metrics['metric/note/recall_frame']) / iteration)
            metrics['metric/note/f1_frame'].append(sum(temp_metrics['metric/note/f1_frame']) / iteration)

        # if save_path is not None:
            # os.makedirs(save_path, exist_ok=True)
            # file_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.file.png')
            # save_pianoroll(file_path, val_data['technique'])
            # pred_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.pred.png')
            # save_pianoroll(pred_path, pred['technique'])
    
    ############ get the macro recall and precision of techniques ############ 
    if eval_tech:
        macro_recall, macro_precision  = get_prec_recall(macro_cm)
        for key, value in technique_dict.items():
            # if key == 0:
            #     continue
            p = macro_precision[key][key] #[key-1][key-1]
            r = macro_recall[key][key]
            f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
            metrics[f'metric/{value}/macro_precision'].append(p)
            metrics[f'metric/{value}/macro_recall'].append(r)
            metrics[f'metric/{value}/macro_f1'].append(f)
        cm_dict_all = {}
        cm_dict_all['cm'] = macro_cm
        cm_dict_all['Precision'] = macro_precision
        cm_dict_all['Recall'] = macro_recall

        macro_tech_label = torch.tensor(macro_tech_label)
        macro_tech_pred = torch.tensor(macro_tech_pred)
        tech_ref, tech_i_ref = extract_technique(macro_tech_label, scale2time=True) # (tech_label, state_label)
        tech_est, tech_i_est = extract_technique(macro_tech_pred, scale2time=True) # (pred['tech'], pred['note_state'])
        metrics = evaluate_technique(tech_ref, tech_i_ref, tech_est, tech_i_est, technique_dict, metrics, macro=True)
    ############ get macro note metrics ############
    if eval_note:
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