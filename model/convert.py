from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def get_confusion_matrix(correct_labels, predict_labels):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)

    # recall
    cm_recall = []
    for i, row in enumerate(cm):
        total = row.sum()
        for j in range(cm.shape[0]):
            cm_recall.append(row[j]/total if (row[j] != 0 or total != 0) else 0)
    cm_recall = np.reshape(cm_recall, cm.shape)

    # precision
    cm_precision = []
    for j, col in enumerate(cm.T):
        total = col.sum()
        for i in range(cm.shape[0]):
            cm_precision.append(col[i] / total if (col[j] != 0 or total != 0) else 0)
    cm_precision = np.reshape(cm_precision, cm.shape).T

    return cm, cm_recall, cm_precision

def evaluate_frame_accuracy(correct_labels, predict_labels):
    matched = 0
    for ref, est in zip(correct_labels, predict_labels):
        if ref == est:
            matched += 1
    return matched / len(correct_labels)

def evaluate_frame_accuracy_per_tech(tech, correct_labels, predict_labels):
    
    accuracy_per_tech = torch.zeros(10, 2) # [[# note, # ac note], [], ... ]
    accuracy = torch.zeros(10)
    for tech, ref, est in zip(tech, correct_labels, predict_labels):
        accuracy_per_tech[tech][0] += 1
        if ref == est:
            accuracy_per_tech[tech][1] += 1
    
    for i in range(len(accuracy_per_tech)):
        # accuracy
        accuracy[i] = accuracy_per_tech[i][1] / accuracy_per_tech[i][0]
    
    return accuracy

def extract_technique(tech, state=None):
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
    tech = tech.to(torch.int8).cpu() # float

    techniques = []
    intervals = []

    # get the interval of every technique
    i = 0
    if state is None:
        while i < len(tech):
            technique = tech[i]

            onset = i
            offset = i
            while offset < len(tech) and tech[offset] == technique:
                offset += 1
            # After knowing where does the note start and end, we can return the technique information
            techniques.append(technique)
            intervals.append([onset, offset - 0.1]) # offset - 1
            i = offset
    else:
        while i < len(tech):
            if state[i] == 1: # tech onset at frame i
                onset_tech = tech[i]
                onset = i
                i += 1
                while i < len(tech) and state[i] != 1:
                    i += 1
                techniques.append(onset_tech)
                intervals.append([onset, i - 0.1])          
            else:
                i += 1

    return np.array(techniques), np.array(intervals)


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

def extract_notes(notes):
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

    # get the interval of every note
    i = 0
    while i < len(notes):
        note = notes[i]
        onset = i
        offset = i

        while offset < len(notes) and notes[offset] == note:
            offset += 1
        # After knowing where does the note start and end, we can return the note information
        pitches.append(note)
        intervals.append([onset, offset - 0.1]) # offset - 1
        i = offset

    return np.array(pitches), np.array(intervals)