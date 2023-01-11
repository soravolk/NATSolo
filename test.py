import os
import argparse
import torch
import numpy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.tensorboard import SummaryWriter
from model.UNet import UNet
from model.dataset import Solo
from model.convert import *
from model.evaluate_functions import *
from model.utils import plot_confusion_matrix

def parse():
    parser = argparse.ArgumentParser(description='EG-Solo testing script')
    parser.add_argument('model', type=str, help='Path of the transcription model')
    parser.add_argument('--dataset', type=str, default='EG_Solo', help='dataset to evaluate on')
    parser.add_argument('--state', type=int, default=1, help='whether to predict states')
    parser.add_argument('--group', type=int, default=1, help='whether to predict groups')
    parser.add_argument('--note', type=int, default=1, help='whether to predict notes')
    parser.add_argument('--tech', type=int, default=1, help='whether to predict techniques')
    parser.add_argument('--save_path', type=str, default='testing', help='Path to save midi, transcriptions and corresponding metrics')
    parser.add_argument('--audio_type', type=str, default='flac', help='Path of the transcription model')
    return parser

def plot_predicted_transcription(specs, probs, note_interval, note, tech_interval, tech, state, save_folder):
    tech_trans = {
        0: '0',
        1: 's',
        2: 'b',
        3: '~',
        4: 'x',
        5: 'p',
        6: '<>',
        7: 'h',
        8: 't',
        9: 'n'
    }
    # specs = specs.cpu().detach().numpy()
    for i, (spec, prob, x, y, x_tech, y_tech, onset) in enumerate(zip(specs, probs, note_interval, note, tech_interval, tech, state)):
        time_range = spec.shape[0] * (HOP_LENGTH/SAMPLE_RATE)
        fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(100,20), gridspec_kw={'height_ratios': [4,1]})
        ax = ax.flat
        # spectrogram
        # librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        # ax[0].set_ylabel('f (Hz)', fontsize=70)
        # ax[0].tick_params(labelsize=65)
        # label_format = '{:,.0f}'
        # ticks_loc = ax[0].get_yticks().tolist()
        # ax[0].set_ylim([512, 4096])
        # ax[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # ax[0].set_yticklabels([label_format.format(x) for x in ticks_loc])
        # # note prob
        # ax[1].imshow(np.flip(prob[0].transpose(), 0), cmap='plasma')
        # ax[1].set_xlim([0, spec.shape[0]])
        # ax[1].axis('off')
        # # note prob refined by state
        # ax[2].imshow(np.flip(prob[1].transpose(), 0), cmap='plasma')
        # ax[2].set_xlim([0, spec.shape[0]])
        # ax[2].axis('off')
        # note transcription
        ax[0].tick_params(labelbottom=False, labelsize=65)
        ax[0].set_ylabel('midi #', fontsize=70)
        ax[0].set_xlim([0, time_range])
        ax[0].set_ylim([60, 100])
        # ax[3].set_yticklabels([70, 80, 90])
        label_format = '{:,.0f}'
        ticks_loc = ax[0].get_yticks().tolist()
        ax[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax[0].set_yticklabels([label_format.format(x) for x in ticks_loc])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[0].plot(x_val, y_val, linewidth=7.5)
            # ax[1].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # onset information
        for o in onset:
            ax[0].vlines(o, ymin=60, ymax=100, linestyles='dotted')
        # techique transcription
        ax[1].set_xlabel('time (s)', fontsize=70)
        # ax[4].set_ylabel('technique', fontsize=37)
        ax[1].tick_params(labelleft=False, labelsize=60)#, labelbottom=False)
        ax[1].set_xlim([0, time_range])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[1].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=45)
            ax[1].plot(x_val, y_val, linewidth=7.5)
            ax[1].vlines(t[0], ymin=0.7, ymax=2, linestyles='dotted')
        plt.savefig(f'{save_folder}/prediction/{i}.png')
        plt.close()

def plot_groundtruth_transcription(specs, note_interval, note, tech_interval, tech, state, save_folder):
    tech_trans = {
        0: '0',
        1: 's',
        2: 'b',
        3: '~',
        4: 'x',
        5: 'p',
        6: '<>',
        7: 'h',
        8: 't',
        9: 'n'
    }
    # specs = specs.cpu().detach().numpy()
    for i, (spec, x, y, x_tech, y_tech, onset) in enumerate(zip(specs, note_interval, note, tech_interval, tech, state)):
        time_range = spec.shape[0] * (HOP_LENGTH/SAMPLE_RATE)
        fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(100,20), gridspec_kw={'height_ratios': [4, 1]})
        ax = ax.flat
        # spectrogram
        # librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        # ax[0].set_ylabel('spectrogram')
        # note transcription
        ax[0].tick_params(labelbottom=False, labelsize=65)
        ax[0].set_ylabel('midi # (GT)', fontsize=70)
        ax[0].set_xlim([0, time_range])
        ax[0].set_ylim([60, 100])
        label_format = '{:,.0f}'
        ticks_loc = ax[0].get_yticks().tolist()
        ax[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax[0].set_yticklabels([label_format.format(x) for x in ticks_loc])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[0].plot(x_val, y_val, linewidth=7.5)
            # ax[0].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # onset information
        for o in onset:
            ax[0].vlines(o, ymin=60, ymax=100, linestyles='dotted')
        # techique transcription
        ax[1].tick_params(labelleft=False, labelsize=60)
        ax[1].set_xlabel('time (s)', fontsize=70)
        # ax[1].set_ylabel('technique', fontsize=60)
        ax[1].set_xlim([0, time_range])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[1].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=45)
            ax[1].plot(x_val, y_val, linewidth=7.5)
            ax[1].vlines(t[0], ymin=0.7, ymax=2, linestyles='dotted')
        plt.savefig(f'{save_folder}/groundtruth/{i}.png')
        plt.close()

def save_transcription_and_midi(inferences, spec, prob, save_folder, scaling):
    transcription_path = f'{save_folder}/transcription'
    midi_path = f'{save_folder}/midi'

    if not os.path.isdir(transcription_path):
        os.mkdir(transcription_path)
    if not os.path.isdir(f"{transcription_path}/groundtruth"):
        os.mkdir(f"{transcription_path}/groundtruth")
    if not os.path.isdir(f"{transcription_path}/prediction"):
        os.mkdir(f"{transcription_path}/prediction")

    if not os.path.isdir(midi_path):
        os.mkdir(midi_path)

    transcriptions = defaultdict(list)
    note_labels = inferences['testing_note_label']
    state_labels = inferences['testing_state_label']
    tech_labels = inferences['testing_tech_label']
    note_preds = inferences['testing_note_pred']
    state_preds = inferences['testing_state_pred']
    tech_preds = inferences['testing_tech_pred']
    ep = 1
    for i, (s_label, s_pred, n_label, n_pred, t_label, t_pred) in enumerate(zip(state_labels, state_preds, note_labels, note_preds, tech_labels, tech_preds)):
        transcriptions = gen_transcriptions_and_midi(transcriptions, s_label, s_pred, n_label, n_pred, t_label, t_pred, i, ep, midi_path, scaling=scaling)

    plot_groundtruth_transcription(spec, transcriptions['note_interval_gt'], transcriptions['note_gt'], transcriptions['tech_interval_gt'], transcriptions['tech_gt'], transcriptions['state_gt'], transcription_path)
    plot_predicted_transcription(spec, prob, transcriptions['note_interval'], transcriptions['note'], transcriptions['tech_interval'], transcriptions['tech'], transcriptions['state'], transcription_path)

def save_confusion_mat(cm_dict_all, save_folder):
    cmat_path = f'{save_folder}/confusion_matrix'
    ep = 1
    for output_key in ['cm', 'Recall', 'Precision']:
        if output_key == 'cm':
            plot_confusion_matrix(cm_dict_all[output_key], None, ep, output_key, f'{output_key}_all', 'd', 10, cmat_path)
        else:
            plot_confusion_matrix(cm_dict_all[output_key], None, ep, output_key, f'{output_key}_all', '.2f', 6, cmat_path)

def load_data(audio_type, device, name):
    if name == 'EG_Solo':
        return Solo(path='./Solo', folders=['valid'], sequence_length=None, device=device, audio_type=audio_type)
    elif name == 'GN':
        return Solo(path='./GN', folders=['valid'], sequence_length=None, device=device, audio_type=audio_type)

def load_model(path, device):
    ds_ksize, ds_stride = 2, 2
    # eps=1.3 #2 
    model = UNet(ds_ksize, ds_stride, log=True, spec='Mel', device=device)
    model.to(device)
    model.load_state_dict(torch.load(path))
    return model

def save_metrics(metrics, save_folder, dataset):
    metrics_path = f'{save_folder}/metrics'

    if not os.path.isdir(metrics_path):
        os.mkdir(metrics_path)

    if not os.path.isdir(f"{metrics_path}/{dataset}"):
        os.mkdir(f"{metrics_path}/{dataset}")

    with open(f'{metrics_path}/{dataset}/metrics.txt', 'w') as f:
        for key, values in metrics.items():
            if key.startswith('metric/'):
                _, category, name = key.split('/')
                # show metrics on terminal
                print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}', file=f)
    # for key, values in metrics.items():
    #     if key.startswith('metric/'):
    #         writer.add_scalar(key, np.mean(values), global_step=ep)

def evaluation(model, testing_set, save_path, has_state, has_group, has_note, has_tech, dataset):

    scaling = HOP_LENGTH / SAMPLE_RATE
    ep = 1
    model.eval()
    with torch.no_grad():
        metrics, cm_dict_all, inferences, spec, prob = evaluate_prediction(testing_set, model, ep, scaling, save_path=save_path, testing=True, has_state=has_state, has_group=has_group, eval_note=has_note, eval_tech=has_tech)
        save_metrics(metrics, args.save_path, dataset)

        if has_note and has_tech:
            save_transcription_and_midi(inferences, spec, prob, args.save_path, scaling)
        # if has_tech:
        #     save_confusion_mat(cm_dict_all, args.save_path)
    
if __name__ == '__main__':
    device = 'cuda:0'
    parser = parse()
    args = parser.parse_args()
    state = (args.state == 1)
    group = (args.group == 1)
    note = (args.note == 1)
    tech = (args.tech == 1)
    testing_set = load_data(args.audio_type, device, args.dataset)
    model = load_model(args.model, device)

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    evaluation(model, testing_set, args.save_path, state, group, note, tech, args.dataset)
    # for dataset in ['EG_Solo', 'GN']:
    #     testing_set = load_data(args.audio_type, device, dataset)
    #     writer = SummaryWriter(f'runs/SOTA-Ablation-S{state}-G{group}-N{note}-T{tech}-no-refine-on-{dataset}')
    #     for ep in range(2100, 8000, 100):
    #         model = load_model(f'{args.model}/model-{ep}.pt', device)
    #         evaluation(model, testing_set, args.save_path, state, group, note, tech, writer, ep)
    
    