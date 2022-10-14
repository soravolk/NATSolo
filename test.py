import argparse
import torch
import numpy
import librosa
import librosa.display
import matplotlib.pyplot as plt
from model.UNet import UNet
# from onsets_and_frames.onsets_and_frames import *
# from onsets_and_frames.onsets_and_frames.transcriber import OnsetsAndFrames
from onsets_and_frames.evaluate import evaluate
from onsets_and_frames.onsets_and_frames.dataset import Solo as Solo_OF
from model.dataset import Solo
from model.convert import *
from model.evaluate_functions import *
from model.utils import plot_confusion_matrix

def parse():
    parser = argparse.ArgumentParser(description='testing script')
    parser.add_argument('model', type=str, help='Path of the transcription model')
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model')
    parser.add_argument('--note', type=bool, default=True, help='whether to evaluate notes')
    parser.add_argument('--tech', type=bool, default=True, help='whether to evaluate techniques')
    parser.add_argument('--save_path', type=str, default='testing', help='Path to save midi, transcriptions and corresponding metrics')
    parser.add_argument('--audio_type', type=str, default='flac', help='Path of the transcription model')
    return parser

def load_data(audio_type, device, name):
    if name == 'unet':
        return Solo(folders=['valid'], sequence_length=None, device=device, audio_type=audio_type)
    elif name == 'of':
        return Solo_OF(path='./Solo', folders=['valid'], sequence_length=None, device=device, audio_type=audio_type)

def load_model(path, device, name):
    ds_ksize, ds_stride = 2, 2
    # eps=1.3 #2 
    if name == 'unet':
        model = UNet(ds_ksize, ds_stride, log=True, spec='Mel', device=device)
        model.to(device)
        model.load_state_dict(torch.load(path))
    elif name == 'of':
        # model = OnsetsAndFrames(229, MAX_MIDI - MIN_MIDI + 1, 48)
        # print(model)
        model = torch.load(path)
    return model

def save_metrics(metrics, save_folder):
    metrics_path = f'{save_folder}/metrics'
    with open(f'{metrics_path}/metrics.txt', 'w') as f:
        for key, values in metrics.items():
            if key.startswith('metric/'):
                _, category, name = key.split('/')
                # show metrics on terminal
                print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}', file=f)

def plot_predicted_transcription(specs, probs, note_interval, note, tech_interval, tech, state, save_folder):
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
    # specs = specs.cpu().detach().numpy()
    for i, (spec, prob, x, y, x_tech, y_tech, onset) in enumerate(zip(specs, probs, note_interval, note, tech_interval, tech, state)):
        time_range = spec.shape[0] * (HOP_LENGTH/SAMPLE_RATE)
        fig, ax = plt.subplots(5, 1, constrained_layout=True, figsize=(48,20), gridspec_kw={'height_ratios': [4,2,2,4,1]})
        ax = ax.flat
        # spectrogram
        librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        ax[0].set_ylabel('spectrogram')
        # note prob
        ax[1].imshow(np.flip(prob[0].transpose(), 0), cmap='plasma')
        ax[1].axis('off')
        # note prob refined by state
        ax[2].imshow(np.flip(prob[1].transpose(), 0), cmap='plasma')
        ax[2].axis('off')
        # note transcription
        ax[3].tick_params(labelbottom=False)
        ax[3].set_ylabel('midi note numbers')
        ax[3].set_xlim([0, time_range])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[3].plot(x_val, y_val)
            # ax[1].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # onset information
        for o in onset:
            ax[3].vlines(o, ymin=51, ymax=100, linestyles='dotted')
        # techique transcription
        ax[4].set_xlabel('time (t)')
        ax[4].set_ylabel('technique')
        ax[4].set_xlim([0, time_range])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[4].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=35)
            ax[4].plot(x_val, y_val)
            ax[4].vlines(t[0], ymin=0, ymax=9, linestyles='dotted')
        plt.savefig(f'{save_folder}/prediction/{i}.png')
        plt.close()

def plot_groundtruth_transcription(specs, note_interval, note, tech_interval, tech, state, save_folder):
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
    # specs = specs.cpu().detach().numpy()
    for i, (spec, x, y, x_tech, y_tech, onset) in enumerate(zip(specs, note_interval, note, tech_interval, tech, state)):
        # print('++++++++++++++++++++', spec.shape)
        time_range = spec.shape[0] * (HOP_LENGTH/SAMPLE_RATE)
        fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(48,20), gridspec_kw={'height_ratios': [4,4,1]})
        ax = ax.flat
        # spectrogram
        librosa.display.specshow(spec.transpose(), y_axis='mel', sr=SAMPLE_RATE, fmax=MEL_FMAX, ax=ax[0])
        ax[0].set_ylabel('spectrogram')
        # note transcription
        ax[1].tick_params(labelbottom=False)
        ax[1].set_ylabel('midi note numbers')
        ax[1].set_xlim([0, time_range])
        for j, t in enumerate(x):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            y_val = np.full(len(x_val), y[j])
            ax[1].plot(x_val, y_val)
            # ax[1].vlines(t[0], ymin=51, ymax=100, linestyles='dotted')
        # onset information
        for o in onset:
            ax[1].vlines(o, ymin=51, ymax=100, linestyles='dotted')
        # techique transcription
        ax[2].set_xlabel('time (t)')
        ax[2].set_ylabel('technique')
        ax[2].set_xlim([0, time_range])
        for j, t in enumerate(x_tech):
            x_val = np.arange(t[0], t[1], HOP_LENGTH/SAMPLE_RATE)
            # y_val = np.full(len(x_val), y_tech[j])
            y_val = np.ones(len(x_val))
            ax[2].text(x_val[len(x_val) // 2], 1, tech_trans[y_tech[j]], fontsize=35)
            ax[2].plot(x_val, y_val)
            ax[2].vlines(t[0], ymin=0, ymax=9, linestyles='dotted')
        plt.savefig(f'{save_folder}/groundtruth/{i}.png')
        plt.close()

def save_transcription_and_midi(inferences, spec, prob, save_folder):
    transcription_path = f'{save_folder}/transcription'
    midi_path = f'{save_folder}/midi'

    transcriptions = defaultdict(list)
    note_labels = inferences['testing_note_label']
    state_labels = inferences['testing_state_label']
    tech_labels = inferences['testing_tech_label']
    note_preds = inferences['testing_note_pred']
    state_preds = inferences['testing_state_pred']
    tech_preds = inferences['testing_tech_pred']
    ep = 1
    for i, (s_label, s_pred, n_label, n_pred, t_label, t_pred) in enumerate(zip(state_labels, state_preds, note_labels, note_preds, tech_labels, tech_preds)):
        transcriptions = gen_transcriptions_and_midi(transcriptions, s_label, s_pred, n_label, n_pred, t_label, t_pred, i, ep, midi_path)

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

def evaluation(model, testing_set, save_path, eval_note, eval_tech, name):
    technique_dict = {
        0: 'no tech',
        1: 'slide',
        2: 'bend',
        3: 'trill',
        4: 'mute',
        5: 'pull',
        6: 'harmonic',
        7: 'hammer',
        8: 'tap'
    }
    # model.eval()
    # with torch.no_grad():
    #     metrics = evaluate_wo_prediction(tqdm(testing_set), model, reconstruction=False,
    #                                    save_path=os.path.join(logdir,'./MIDI_results'))
        
    # for key, values in metrics.items():
    #     if key.startswith('metric/'):
    #         _, category, name = key.split('/')
    #         print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
         
    # export_path = os.path.join(logdir, 'result_dict')    
    # pickle.dump(metrics, open(export_path, 'wb'))

    ep = 1
    model.eval()
    with torch.no_grad():
        if name == 'unet':
            metrics, cm_dict_all, inferences, spec, prob = evaluate_prediction(testing_set, model, ep, technique_dict, save_path=save_path, testing=True, eval_not=eval_note, eval_tech=eval_tech)
        elif name == 'of':
            metrics = evaluate(testing_set, model)
        save_metrics(metrics, args.save_path)

        if eval_note and eval_tech:
            save_transcription_and_midi(inferences, spec, prob, args.save_path)
        if eval_tech:
            save_confusion_mat(cm_dict_all, args.save_path)
    
if __name__ == '__main__':
    device = 'cuda:0'
    parser = parse()
    args = parser.parse_args()
    testing_set = load_data(args.audio_type, device, args.model_name)
    model = load_model(args.model, device, args.model_name)
    evaluation(model, testing_set, args.save_path, args.note, args.tech, args.model_name)
    