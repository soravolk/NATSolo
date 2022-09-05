from sklearn.metrics import confusion_matrix
from mido import Message, MidiFile, MidiTrack
import os
from mir_eval.util import hz_to_midi, midi_to_hz
from collections import defaultdict
import numpy as np
import torch
from .constants import *

def state2time(states):
    states = states.to(torch.int8).cpu()
    scaling = HOP_LENGTH / SAMPLE_RATE
    onset_time = []
    for i, s in enumerate(states):
        if s == 1:
            if i + 1 < len(states) and states[i + 1] != 1:
                onset_time.append(i)
    return np.array(onset_time) * scaling

def get_transcription_and_cmx(labels, preds, ep, technique_dict):
    transcriptions = defaultdict(list)

    for i, (label, tech, note, note_state) in enumerate(zip(labels, preds['tech'], preds['note'], preds['note_state'])):
        # get label from one hot vectors
        state_label = label[:,:3].argmax(axis=1)
        note_label = label[:,7:57].argmax(axis=1)
        tech_label = label[:,57:].argmax(axis=1)    
        
        tech_ref, tech_i_ref = extract_technique(tech_label) # (tech_label, state_label)
        tech_est, tech_i_est = extract_technique(tech.squeeze(0))

        midi_path = os.path.join('midi', f'song{i}_ep{ep}.midi')
        gt_midi_path = os.path.join('midi', f'gt_song{i}.midi')
        note_ref, note_i_ref = extract_notes(note_label, midi=True, path=gt_midi_path)
        note_est, note_i_est = extract_notes(note, midi=True, path=midi_path)

        transcriptions['tech_gt'].append(tech_ref)
        transcriptions['tech_interval_gt'].append(tech_i_ref)
        transcriptions['note_gt'].append(note_ref + LOGIC_MIDI)
        transcriptions['note_interval_gt'].append(note_i_ref)
        transcriptions['state_gt'].append(state2time(state_label))
        transcriptions['tech'].append(tech_est)
        transcriptions['tech_interval'].append(tech_i_est)
        transcriptions['note'].append(note_est + LOGIC_MIDI)
        transcriptions['note_interval'].append(note_i_est)
        transcriptions['state'].append(state2time(note_state))
    
    # get macro metrics
    tech_label = labels[:,:,57:].argmax(axis=2).flatten()
    tech_pred = preds['tech'].flatten()
    cm_dict = get_confusion_matrix(tech_label.cpu().numpy(), tech_pred.cpu().numpy(), list(technique_dict.keys()))

    return transcriptions, cm_dict

def get_prec_recall(cm):
    # recall
    cm_recall = []
    for i, row in enumerate(cm):
        total = row.sum()
        for j in range(cm.shape[0]):
            cm_recall.append(row[j] / total if (row[j] != 0 or total != 0) else 0)
    cm_recall = np.reshape(cm_recall, cm.shape)

    # precision
    cm_precision = []
    for j, col in enumerate(cm.T):
        total = col.sum()
        for i in range(cm.shape[0]):
            cm_precision.append(col[i] / total if (col[j] != 0 or total != 0) else 0)
    cm_precision = np.reshape(cm_precision, cm.shape).T
    return cm_recall, cm_precision

def get_confusion_matrix(correct_labels, predict_labels, labels):
    """
    Generate confusion matrix, recall and precision
    Parameters
    ----------
    correct_labels: 
    predict_labels: 
    Returns
    -------
    cm: 
    cm_recall:
    cm_precision:
    """
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    cm_new = []
    for i in range(len(cm)):
        if i == 0:
            continue
        cm_new.append(cm[i][1:])
    cm_new = np.reshape(cm_new, (len(labels)-1, len(labels)-1))

    cm_recall, cm_precision = get_prec_recall(cm_new)

    cm_dict = {
        'cm': cm_new,
        'Precision': cm_precision,
        'Recall': cm_recall,
    }

    return cm_dict

def evaluate_frame_accuracy(correct_labels, predict_labels):
    matched = 0
    for ref, est in zip(correct_labels, predict_labels):
        if ref == est:
            matched += 1
    return matched / len(correct_labels)

def evaluate_frame_accuracy_per_tech(tech, correct_labels, predict_labels):
    
    accuracy_per_tech = torch.zeros(9, 2) # [[# note, # ac note], [], ... ]
    accuracy = torch.zeros(9)
    for tech, ref, est in zip(tech, correct_labels, predict_labels):
        accuracy_per_tech[tech][0] += 1
        if ref == est:
            accuracy_per_tech[tech][1] += 1
    
    for i in range(len(accuracy_per_tech)):
        # accuracy
        accuracy[i] = accuracy_per_tech[i][1] / accuracy_per_tech[i][0]
    
    return accuracy

def extract_technique(techs, states=None, scale2time=True):
    """
    Finds the note timings based on the onsets and tech information
    Parameters
    ----------
    tech: torch.FloatTensor, shape = [tech, bins]
    gt: if the input tech is groundtruth
    Returns
    -------
    techniques: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    # onsets = onsets.cpu().to(torch.uint8)
    # onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step
    
    # convert from label 0 - 9 to 1 - 10 for mir_eval by adding 1
    techs = techs.to(torch.int8).cpu() # float

    techniques = []
    intervals = []

    # get the interval of every technique
    i = 0
    rb = len(techs)
    if states is None:
        while i < rb:
            tech = techs[i]
            if tech == 0:
                i += 1
                continue
            onset = i
            offset = i
            while offset < rb and techs[offset] == tech:
                offset += 1
            # After knowing where does the note start and end, we can return the tech information
            techniques.append(tech)
            intervals.append([onset, offset - 0.1]) # offset - 1
            i = offset
    else:
        states = states.to(torch.int8).cpu()
        # a valid note must come with the onset state
        while i < rb:
            if tech[i] != 0 and states[i] == 1: # tech onset at frame i:
                onset = i
                onset_tech = tech[i]
                while i + 1 < rb and states[i + 1] == 1:
                    i += 1
                i += 1
                if i >= rb:
                    break
                if states[i] == 0 or onset_tech == 0:
                    continue

                offset = i
                while offset < rb and states[offset] == 2:
                    offset += 1
                techniques.append(onset_tech)
                intervals.append([onset, offset - 0.1])          
            else:
                i += 1
    # convert time steps to seconds
    scaling = HOP_LENGTH / SAMPLE_RATE
    if scale2time:
        intervals = (np.array(intervals) * scaling).reshape(-1, 2)

    return np.array(techniques), intervals


def techniques_to_frames(techniques, intervals, shape):
    """
    Takes lists specifying technique sequences and return
    Parameters
    ----------
    techniques: list of technique bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_tech, n_bins]
    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for technique, (onset, offset) in zip(techniques, intervals):
        roll[onset:offset, technique] = 1

    time = np.arange(roll.shape[0])
    tehcniques = [roll[t, :].nonzero()[0] for t in time]
    return time, tehcniques

def extract_notes(notes, states=None, groups=None, scale2time=True, midi=False, path=None):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    notes = notes.to(torch.int8).cpu()

    pitches = []
    intervals = []

    if states is None:
        # get the interval of every note
        i = 0
        while i < len(notes):
            note = notes[i]
            if note == 0:
                i += 1
                continue
            onset = i
            offset = i

            while offset < len(notes) and notes[offset] == note:
                offset += 1
            # After knowing where does the note start and end, we can return the note information
            pitches.append(note)
            intervals.append([onset, offset - 0.1]) # offset - 1
            i = offset
    else:
        states = states.to(torch.int8).cpu()
        # a valid note must come with the onset state
        onset = False
        total = len(notes)
        for i, (note, state) in enumerate(zip(notes, states)):
            # assume one onset does not follow another onset
            if onset == False and state == 1:
                onset = True
                start = i
                onset_note = note

            if onset:
                if state != 0:
                    if (i + 1) < total and states[i] == 2 and states[i + 1] == 1:
                        onset = False
                        pitches.append(onset_note)
                        intervals.append([start, i - 0.1])
                    else:
                        continue
                else:
                    onset = False
                    pitches.append(onset_note)
                    intervals.append([start, i - 0.1])

    scaling = HOP_LENGTH / SAMPLE_RATE
    if scale2time and midi:
        intervals = (np.array(intervals) * scaling).reshape(-1, 2)
        save_midi(path, np.array(pitches), intervals)
        pitches = np.array(pitches)
    elif scale2time:
        intervals = (np.array(intervals) * scaling).reshape(-1, 2)
        # converting midi number to frequency
        pitches = np.array([midi_to_hz(MIN_MIDI + midi) for midi in pitches])

    
    return pitches, intervals

def save_midi(path, pitches, intervals):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0
    #pitches = np.array([midi_to_hz(LOGIC_MIDI + midi) for midi in midi_notes])
    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        #pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=event['pitch']+LOGIC_MIDI, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)