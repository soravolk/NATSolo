import sys
from functools import reduce
import numpy as np
import torch
from PIL import Image
from torch.nn.modules.module import _addindent
import itertools
import matplotlib.pyplot as plt
import librosa
import librosa.display
from .constants import *

def cycle(iterable):
    while True:
        for item in iterable:
            yield item

def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params) #[92m is green color, [0m is black color
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

def plot_spec_and_post(writer, ep, source, figname):
    source = source.cpu().detach().numpy()
    fig, axs = plt.subplots(2, 2, figsize=(24,8))
    axs = axs.flat
    for idx, i in enumerate(source):
        axs[idx].imshow(np.flip(i.transpose(), 0))
        axs[idx].axis('off')
    fig.tight_layout()
    writer.add_figure(figname, fig , ep)

def plot_transcription_A(writer, ep, figname, note_interval, note, tech_interval, tech):
    fig = plt.figure(constrained_layout=True, figsize=(48,20))
    subfigs = fig.subfigures(2, 2)
    subfigs = subfigs.flat
    for i, (x, y, x_tech, y_tech) in enumerate(zip(note_interval, note, tech_interval, tech)):
        subfigs[i].suptitle(f'Transcription {i}')
        ax = subfigs[i].subplots(2,1)
        ax = ax.flat
        # note transcription
        ax[0].set_title('Note')
        ax[0].set_xlabel('time (t)')
        ax[0].set_ylabel('midi note numbers')
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], 0.1)
            y_val = np.full(len(x_val), y[j])
            ax[0].plot(x_val, y_val)
            ax[0].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # techique transcription
        ax[1].set_title('Technique')
        ax[1].set_xlabel('time (t)')
        ax[1].set_ylabel('technique')
        ax[1].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['no_tech', 'slide', 'bend', 'trill', 'mute', 'pull', 'harmonic', 'hammer', 'tap'])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], 0.1)
            y_val = np.full(len(x_val), y_tech[j])
            ax[1].plot(x_val, y_val)
            ax[1].vlines(t[0], ymin=0, ymax=8, linestyles='dotted')
    writer.add_figure(figname, fig, ep)

def plot_transcription_B(writer, ep, figname, mel, note_interval, note, tech_interval, tech):
    tech_trans = {
        0: '0',
        1: 's',
        2: 'b',
        3: 'tri',
        4: 'x',
        5: 'p',
        6: 'har',
        7: 'h',
        8: 't'
    }
    mel = mel.cpu().detach().numpy()
    fig = plt.figure(constrained_layout=True, figsize=(48,25))
    subfigs = fig.subfigures(2, 2)
    subfigs = subfigs.flat
    for i, (spec, x, y, x_tech, y_tech) in enumerate(zip(mel, note_interval, note, tech_interval, tech)):
        subfigs[i].suptitle(f'Transcription {i}')
        ax = subfigs[i].subplots(3, 1, gridspec_kw={'height_ratios': [4, 4,1]})
        ax = ax.flat
        # spectrogram
        #S_dB = librosa.power_to_db(spec, ref=np.max)
        librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        #ax[0].imshow(spec.transpose())
        ax[0].set_ylabel('spectrogram')
        #ax[0].set_xlim([0, 163840 // HOP_LENGTH])
        # note transcription
        ax[1].tick_params(labelbottom=False)
        ax[1].set_ylabel('midi note numbers')
        ax[1].set_xlim([0, 6])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[1].plot(x_val, y_val)
            ax[1].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # techique transcription
        ax[2].set_xlabel('time (t)')
        ax[2].set_ylabel('technique')
        ax[2].set_xlim([0, 6])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[2].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=35)

            ax[2].plot(x_val, y_val)
            ax[2].vlines(t[0], ymin=0, ymax=9, linestyles='dotted')
    writer.add_figure(figname+'_B', fig, ep)

def plot_transcription(writer, ep, figname, mel, note_interval, note, tech_interval, tech):
    plot_transcription_A(writer, ep, figname, note_interval, note, tech_interval, tech)
    # another representation of transcription
    plot_transcription_B(writer, ep, figname, mel, note_interval, note, tech_interval, tech)

def plot_confusion_matrix(cm, writer, ep, title='Prediction', tensor_name = 'MyFigure/image', dtype='d', fontsize=20):
    ''' 
    Parameters:
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    technique_dict = {
        1: 'slide',
        2: 'bend',
        3: 'trill',
        4: 'mute',
        5: 'pull',
        6: 'harmonic',
        7: 'hammer',
        8: 'tap'
    }

    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180, facecolor='w', edgecolor='k')

    im = ax.imshow(cm, cmap='Oranges')
    ax.set_title(title)
    fig.colorbar(im)

    classes = list(technique_dict.values())

    tick_marks = np.arange(len(classes))

    # ax = plt.gca()
    ax.set_xlabel('Predicted Technique', fontsize=6)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=5, rotation=45,  ha='center')

    ax.set_ylabel('True Technique', fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=5, va ='center')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], dtype) if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=fontsize, verticalalignment='center', color= "white")
    fig.set_tight_layout(True)
    writer.add_figure(tensor_name, fig , ep)

def flatten_attention(a, w_size=31):
    w_size = (w_size-1)//2 # make it half window size
    seq_len = a.shape[0]
    n_heads = a.shape[1]
    attentions = torch.zeros(seq_len, seq_len)
    for t in range(seq_len):
        start = 0 if t-w_size<0 else t-w_size
        end = seq_len if t+w_size > seq_len else t+w_size
        if t<w_size:
            attentions[t, start:end+1] = a[t, -(end-start)-1:]
        else:
            attentions[t, start:end] = a[t, :(end-start)]
            
    return attentions

def save_pianoroll(path, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram
    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = (1 - (1 - frames)**2)
    image = torch.stack([frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)
    
class Normalization():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""
    def __init__(self, mode='framewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                output = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                output[torch.isnan(output)]=0 # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                return (x-x_min)/(x_max-x_min)
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def transform(self, x):
        return self.normalize(x)