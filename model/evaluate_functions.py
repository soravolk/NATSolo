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

def evaluate_prediction(data, model, ep, technique_dict, save_path=None, reconstruction=True):
    metrics = defaultdict(list) # a safe dict
    macro_cm = None
    macro_note_pred = []
    macro_state_pred = []
    macro_note_label = []
    macro_state_label = []
    val_loss = None
    if len(data) > 5:
        iteration = 1
    else:
        iteration = 5
        
    for val_data in tqdm(data):
        total_loss = 0
        temp_metrics = defaultdict(list)
        for i in range(iteration):
            pred, losses, _, _, _, _ = model.run_on_batch(val_data, None, False)
            # if val_loss is None:
            #     val_loss = defaultdict(list)
            #     for key, value in {**losses}.items():
            #         val_loss[key] = value
            # else:
            #     for key, value in {**losses}.items():
            #         val_loss[key] += value             
            # get label from one hot vector
            ######## for state of size 3 ########
            state_label = val_data['label'][:,:2].argmax(axis=1)
            group_label = val_data['label'][:,2:6].argmax(axis=1)
            note_label = val_data['label'][:,6:56].argmax(axis=1)
            tech_label = val_data['label'][:,56:].argmax(axis=1)
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

            ############ evaluate techniques ############
            # get the confusion matrix
            cm_dict = get_confusion_matrix(tech_label.cpu().numpy(), pred['tech'].squeeze(0).cpu().numpy(), list(technique_dict.keys()))
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

            # find the technique timings
            # groundtruth: val_data['technique'].shape [232]

            pred['note'].squeeze_(0)
            #print(pred['note'])
            pred['note_state'].squeeze_(0)
            if i == 0:
                macro_note_label.extend(note_label)
                macro_state_label.extend(state_label)
                macro_note_pred.extend(pred['note'])
                macro_state_pred.extend(pred['note_state'])
            # get techinique and interval
            tech_ref, tech_i_ref = extract_technique(tech_label, scale2time=True) # (tech_label, state_label)
            tech_est, tech_i_est = extract_technique(pred['tech'].squeeze(0), scale2time=True) # (pred['tech'], pred['note_state'])
            ############ evaluate techniques ############
            # tp, tr, tf, to = evaluate_notes(tech_i_ref, tech_ref, tech_i_est, tech_est, strict=False, offset_ratio=None, pitch_tolerance=0)
            # temp_metrics['metric/tech/precision'].append(tp)
            # temp_metrics['metric/tech/recall'].append(tr)
            # temp_metrics['metric/tech/f1'].append(tf)
            # temp_metrics['metric/tech/overlap'].append(to)

            ############ get note and interval ############ 
            note_ref, note_i_ref = extract_notes(note_label, state_label)
            note_est, note_i_est = extract_notes(pred['note'], pred['note_state'])

            ############ evaluate notes ############
            acc = evaluate_frame_accuracy(note_label, pred['note']) # frame level
            p, r, f, o = evaluate_notes(note_i_ref, note_ref, note_i_est, note_est, strict=False, offset_ratio=None)
            temp_metrics['metric/note/accuracy'].append(acc)
            temp_metrics['metric/note/precision'].append(p)
            temp_metrics['metric/note/recall'].append(r)
            temp_metrics['metric/note/f1'].append(f)
            temp_metrics['metric/note/overlap'].append(o)
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
        for key, value in technique_dict.items():
            metrics[f'metric/{value}/precision'].append(sum(temp_metrics[f'metric/{value}/precision']) / iteration)
            metrics[f'metric/{value}/recall'].append(sum(temp_metrics[f'metric/{value}/recall']) / iteration)
            metrics[f'metric/{value}/f1'].append(sum(temp_metrics[f'metric/{value}/f1']) / iteration)
        metrics['metric/note/accuracy'].append(sum(temp_metrics['metric/note/accuracy']) / iteration)
        metrics['metric/note/precision'].append(sum(temp_metrics['metric/note/precision']) / iteration)
        metrics['metric/note/recall'].append(sum(temp_metrics['metric/note/recall']) / iteration)
        metrics['metric/note/f1'].append(sum(temp_metrics['metric/note/f1']) / iteration)
        metrics['metric/note/overlap'].append(sum(temp_metrics['metric/note/overlap']) / iteration)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.file.png')
            save_pianoroll(file_path, val_data['technique'])
            pred_path = os.path.join(save_path, os.path.basename(val_data['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['technique'])
    
    ############ get the macro recall and precision of technique s############ 
    macro_precision, macro_recall = get_prec_recall(macro_cm)
    for key, value in technique_dict.items():
        # if key == 0:
        #     continue
        p = macro_precision[key][key] #[key-1][key-1]
        r = macro_recall[key][key]
        f = (2 * p * r) / float(p + r) if (p != 0 or r != 0) else 0
        metrics[f'metric/{value}/macro_precision'].append(p)
        metrics[f'metric/{value}/macro_recall'].append(r)
        metrics[f'metric/{value}/macro_f1'].append(f)
    ############ get macro note metrics ############
    macro_note_label = torch.tensor(macro_note_label)
    macro_note_pred = torch.tensor(macro_note_pred)
    macro_note_ref, macro_note_i_ref = extract_notes(macro_note_label, torch.tensor(macro_state_label))
    macro_note_est, macro_note_i_est = extract_notes(macro_note_pred, torch.tensor(macro_state_pred))
    acc = evaluate_frame_accuracy(macro_note_label, macro_note_pred) # frame level
    p, r, f, o = evaluate_notes(macro_note_i_ref, macro_note_ref, macro_note_i_est, macro_note_est, offset_ratio=None)
    metrics['metric/note/accuracy_macro'].append(acc)
    metrics['metric/note/precision_macro'].append(p)
    metrics['metric/note/recall_macro'].append(r)
    metrics['metric/note/f1_macro'].append(f)
    
    return metrics#, val_loss

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