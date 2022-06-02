import numpy as np
import torch

def extract_technique(tech, gt=False):
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
    tech = tech.cpu() # float
    tech = (tech + 1).to(torch.uint8)
    print('technique frame + 1: ', tech)

    techniques = []
    intervals = []

    # get the interval of every technique
    i = 0
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