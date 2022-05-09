import numpy as np
import torch

def extract_technique(frames, rule='rule1', gt=False):
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
    techniques: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    # onsets = onsets.cpu().to(torch.uint8)
    # onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step
    
    frames = frames.cpu().to(torch.uint8)
    # convert from label 0 - 9 to 1 - 10 for mir_eval by adding 1
    #print('technique frame size: ', frames.shape)
    #print('technique frame: ', frames)
    if gt is False:
      frames = (np.argmax(frames, axis=1) + 1)
    else:
      frames = (frames + 1).squeeze(1)
    print('technique frame + 1: ', frames)
    techniques = []
    intervals = []

    i = 0
    while i < len(frames):
        technique = frames[i]

        onset = i
        offset = i
        while offset < len(frames) and frames[offset] == technique:
            offset += 1
        # After knowing where does the note start and end, we can return the technique information
        techniques.append(technique)
        intervals.append([onset, offset - 0.1]) # offset - 1
        i = offset

    return np.array(techniques), np.array(intervals)


def techniques_to_frames(techniques, intervals, shape):
    """
    Takes lists specifying notes sequences and return
    Parameters
    ----------
    techniques: list of technique bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]
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