import os
import torch
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils.class_weight import compute_class_weight
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from itertools import cycle

from model.UNet import UNet
from model.dataset import prepare_VAT_dataset, compute_dataset_weight
from model.utils import *
from model.convert import *
from model.evaluate_functions import *
ex = Experiment('train_original')

# parameters for the network
ds_ksize, ds_stride = (2,2),(2,2)
mode = 'imagewise'
sparsity = 2
output_channel = 2
logging_freq = 100 #100
saving_freq = 200

@ex.config
def config():
    root = 'runs'
    device = 'cuda:0'
    log = True
    w_size = 31
    spec = 'Mel'
    resume_iteration = None # 'model-1200'
    train_on = 'Solo'
    n_heads=4
    iteration = 10 # 10
    VAT_start = 0
    alpha = 1
    VAT=False
    XI= 1e-6
    eps=1.3 # 2
    reconstruction = False
    batch_size = 8
    train_batch_size = 8
    val_batch_size = 3
    sequence_length = 163840 # 327680
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')
    epoches = 8000 #8000 # 20000
    step_size_up = 100
    max_lr = 1e-4
    learning_rate = 5e-4
    learning_rate_decay_steps = 1000
    learning_rate_decay_rate = 0.9 #0.98
    clip_gradient_norm = 3
    validation_length = sequence_length
    refresh = False
    #logdir = f'{root}/Unet_Onset-recons={reconstruction}-XI={XI}-eps={eps}-alpha={alpha}-train_on={train_on}-w_size={w_size}-n_heads={n_heads}-lr={learning_rate}-'+ datetime.now().strftime('%y%m%d-%H%M%S')
    logdir = f'{root}/recons={reconstruction}-VAT={VAT}-lr={learning_rate}-'+ datetime.now().strftime('%y%m%d-%H%M%S') + '_ModifyUnet'

def tensorboard_log(batch_visualize, model, valid_set, val_loader, train_set,
                    ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                    VAT, VAT_start, reconstruction, tech_weights=None):
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
    # log various result from the validation audio
    model.eval()

    if ep%logging_freq==0 or ep==1:
        # on valid set
        with torch.no_grad():
            mertics, _ = evaluate_prediction(valid_set, model, ep, technique_dict, reconstruction=reconstruction, tech_weights=tech_weights)
            for key, values in mertics.items():
                if key.startswith('metric/'):
                    _, category, name = key.split('/')
                    # show metrics on terminal
                    print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                    if 'accuracy' in name or 'precision' in name or 'recall' in name or 'f1' in name:
                        writer.add_scalar(key, np.mean(values), global_step=ep)
        # for key, value in {**val_loss}.items():
        #     writer.add_scalar(key, value.item(), global_step=ep) 

        # visualized validation audio
        predictions, _, mel, features = model.run_on_batch(batch_visualize, None, VAT)

        # Show the original transcription and spectrograms
        #loss = sum(losses.values())
        state_group_post = features[0].squeeze(1)
        tech_note_post = features[1].squeeze(1)
        state_feature = features[2].squeeze(1)
        group_feature = features[3].squeeze(1)
        note_feature = features[4].squeeze(1)
        tech_feature = features[5].squeeze(1)
        # get transcriptions and confusion matrix
        transcriptions, cm_dict = get_transcription_and_cmx(batch_visualize['label'], predictions, ep, technique_dict)
        # plot state_group_post
        plot_spec_and_post(writer, ep, state_group_post, 'images/state_group_post')
        # plot tech_note_post
        plot_spec_and_post(writer, ep, tech_note_post, 'images/tech_note_post')

        plot_spec_and_post(writer, ep, state_feature, 'images/state_feature')
        plot_spec_and_post(writer, ep, group_feature, 'images/group_feature')
        plot_spec_and_post(writer, ep, note_feature, 'images/note_feature')
        plot_spec_and_post(writer, ep, tech_feature, 'images/tech_feature')
        # Show the transcription result in validation period
        print('Show the transcription result')
        plot_transcription(writer, ep, 'transcription/ground_truth', mel, transcriptions['note_interval_gt'], transcriptions['note_gt'], transcriptions['tech_interval_gt'], transcriptions['tech_gt'])

        plot_transcription(writer, ep, 'transcription/prediction', mel, transcriptions['note_interval'], transcriptions['note'], transcriptions['tech_interval'], transcriptions['tech'])

        # Plot confusion matrix
        print('Plot confusion matrix')
        for output_key in ['cm', 'Recall', 'Precision', 'cm_2', 'Recall_2', 'Precision_2']:
            if output_key in cm_dict.keys():
                if output_key in ['cm', 'cm_2']:
                    plot_confusion_matrix(cm_dict[output_key], writer, ep, output_key, f'images/{output_key}', 'd', 10)
                else:
                    plot_confusion_matrix(cm_dict[output_key], writer, ep, output_key, f'images/{output_key}', '.2f', 6)    

        # # show adversarial samples    
        # print('adversarial samples')
        # if predictions['r_adv'] is not None: 
        #     fig, axs = plt.subplots(2, 2, figsize=(24,8))
        #     axs = axs.flat
        #     for idx, i in enumerate(mel.cpu().detach().numpy()):
        #         x_adv = i.transpose()+predictions['r_adv'][idx][0].t().cpu().numpy()
        #         axs[idx].imshow(x_adv, vmax=1, vmin=0, cmap='jet', origin='lower')
        #         axs[idx].axis('off')
        #     fig.tight_layout()
        #     writer.add_figure('images/Spec_adv', fig , ep)

        model.eval()
        test_losses = eval_model(model, ep, val_loader, VAT_start, VAT, tech_weights)
        for key, values in test_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)
        # model.eval()
        # test_losses = eval_model(model, ep, supervised_loader, VAT_start, VAT, tech_weights)
        # for key, values in test_losses.items():
        #     if key.startswith('loss/'):
        #         writer.add_scalar(key, np.mean(values), global_step=ep)
    if ep%(2 * logging_freq) == 0:
        # test on training set
        with torch.no_grad():
            mertics, _ = evaluate_prediction(train_set, model, ep, technique_dict, reconstruction=reconstruction, tech_weights=tech_weights)
            for key, values in mertics.items():
                if key.startswith('metric/'):
                    _, _, name = key.split('/')
                    if name in ['accuracy', 'precision', 'recall', 'f1']:
                        writer.add_scalar(f'{key}_train', np.mean(values), global_step=ep)


def train_VAT_model(model, iteration, ep, l_loader, ul_loader, optimizer, scheduler, clip_gradient_norm, alpha, VAT=False, VAT_start=0):
    model.train()
    batch_size = l_loader.batch_size
    total_loss = 0
    l_loader = cycle(l_loader)
    if ul_loader:
        ul_loader = cycle(ul_loader)
    
    metrics = defaultdict(list)
    for i in tqdm(range(iteration)):
        optimizer.zero_grad()
        batch_l = next(l_loader)
        
        if (ep < VAT_start) or (VAT==False):
            predictions, losses = model.run_on_batch(batch_l, None, False)
        else:
            batch_ul = next(ul_loader)
            predictions, losses = model.run_on_batch(batch_l, batch_ul, VAT)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        loss = 0
        # tweak the loss
        # loss = losses(label) + losses(recon) + alpha*(losses['loss/train_LDS_l']+losses['loss/train_LDS_ul'])/2
        # alpha = 1 in the original paper
        for key in losses.keys():
            if key.startswith('loss/train_LDS'):
                loss += alpha*losses[key]/2  # No need to divide by 2 if you have only _l -> ? but you divide both...
            else:
                loss += losses[key]
  
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)
        print(f'Train Epoch: {ep} [{i*batch_size}/{iteration*batch_size}'
                f'({100. * i / iteration:.0f}%)]'
                f"\tMain Loss: {sum(losses.values()):.6f}\t"
#                 + f"".join([f"{k.split('/')[-1]}={v.item():.3e}\t" for k,v in losses.items()])
                , end='\r') 
    print(' '*100, end = '\r')          
    print(f'Train Epoch: {ep}\tLoss: {total_loss/iteration:.6f}')
    return predictions, losses, metrics, optimizer

@ex.automain
def train(spec, resume_iteration, batch_size, sequence_length, w_size, n_heads, train_batch_size, val_batch_size,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, alpha,
          clip_gradient_norm, refresh, device, epoches, logdir, log, iteration, VAT_start, VAT, XI, eps,
          reconstruction): 
    print_config(ex.current_run)
    # flac for 16K audio
    train_set, unsupervised_set, valid_set = prepare_VAT_dataset(
                                                                          sequence_length=sequence_length,
                                                                          validation_length=sequence_length,
                                                                          refresh=refresh,
                                                                          device=device,
                                                                          audio_type='flac')  
    if VAT:
        unsupervised_loader = DataLoader(unsupervised_set, batch_size, shuffle=True, drop_last=True)
#     train_set, unsupervised_set = torch.utils.data.random_split(dataset, [100, 39],
#                                                                      generator=torch.Generator().manual_seed(42))
    
    # get weight of tech label for BCE loss

    # tech_weights = compute_dataset_weight(device)
    tech_weights = None

    print("train_set: ", len(train_set))
    print("unsupervised_set: ", len(unsupervised_set))
    print("valid_set: ", len(valid_set))
    # print("test_set: ", len(test_set))
    supervised_loader = DataLoader(train_set, train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_set, val_batch_size, shuffle=False, drop_last=True)
    batch_visualize = next(iter(val_loader)) # Getting one fixed batch for visualization   

    # model setting
    ds_ksize, ds_stride = 2, 2   
    model = UNet(ds_ksize,ds_stride, log=log, reconstruction=reconstruction,
                    mode=mode, spec=spec, device=device, XI=XI, eps=eps, weights=tech_weights)
    model.to(device)
    if resume_iteration is None:  
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else: # Loading checkpoints and continue training
        model_path = os.path.join('checkpoint', f'{resume_iteration}.pt')
        model.load_state_dict(torch.load(model_path))
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join('checkpoint', 'last-optimizer-state.pt')))

    summary(model)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,cycle_momentum=False)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    for ep in tqdm(range(1, epoches+1)):
        if VAT==True:
            predictions, losses, train_losses, optimizer = train_VAT_model(model, iteration, ep, supervised_loader, unsupervised_loader,
                                                             optimizer, scheduler, clip_gradient_norm, alpha, VAT, VAT_start)
        else:
            predictions, losses, train_losses, optimizer = train_VAT_model(model, iteration, ep, supervised_loader, None,
                                                             optimizer, scheduler, clip_gradient_norm, alpha, VAT, VAT_start)            
        loss = sum(losses.values())

        # Logging results to tensorboard
        if ep == 1:
            writer = SummaryWriter(logdir) # create tensorboard logger     
        if ep < VAT_start or VAT == False:
            tensorboard_log(batch_visualize, model, valid_set, val_loader, train_set,
                            ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                            False, VAT_start, reconstruction, tech_weights)
        else:
            tensorboard_log(batch_visualize, model, valid_set, val_loader, train_set,
                            ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                            True, VAT_start, reconstruction, tech_weights)            

        # Saving model
        # if (ep)%saving_freq == 0:
        #     torch.save(model.state_dict(), os.path.join('checkpoint', f'model-{ep}.pt'))
        #     torch.save(optimizer.state_dict(), os.path.join('checkpoint', 'last-optimizer-state.pt'))
        
        for key, values in train_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)

        # for key, value in {**losses}.items():
        #     writer.add_scalar(key, value.item(), global_step=ep) 

    """
    # Evaluating model performance on the full MAPS songs in the test split     
    print('Training finished, now evaluating on the MAPS test split (full songs)')
    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_prediction(tqdm(full_validation), model, reconstruction=False,
                                       save_path=os.path.join(logdir,'./MIDI_results'))
        
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
         
    export_path = os.path.join(logdir, 'result_dict')    
    pickle.dump(metrics, open(export_path, 'wb'))
    """
