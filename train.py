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
import torch.nn as nn
from tqdm import tqdm
from itertools import cycle

from model.UNet import UNet
from model.dataset import prepare_dataset, compute_dataset_weight
from model.utils import *
from model.convert import *
from model.evaluate_functions import *
ex = Experiment('train_original')

# parameters for the network
ds_ksize, ds_stride = (2,2),(2,2)
mode = 'imagewise'
sparsity = 2
output_channel = 2
logging_freq = 50 #100
saving_freq = 200

@ex.config
def config():
    root = 'runs'
    # device = 'cuda:0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = True
    w_size = 31
    spec = 'Mel'
    resume_iteration = None # 'model-1200'
    train_on = 'Solo'
    n_heads=4
    iteration = 10 # 10
    alpha = 1
    XI= 1e-6
    eps=1.3 # 2
    reconstruction = False
    train_batch_size = 8
    val_batch_size = 4
    sequence_length = 96000 #163840 # 327680
    # if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
    #     batch_size //= 2
    #     sequence_length //= 2
    #     print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')
    epoches = 2300 #8000 # 20000
    step_size_up = 100
    max_lr = 1e-4
    learning_rate = 5e-4
    learning_rate_decay_steps = 1000
    learning_rate_decay_rate = 0.9 #0.98
    weight_decay = 1e-5
    clip_gradient_norm = 3
    validation_length = sequence_length
    refresh = False
    has_note = True
    has_tech = True
    has_state = True
    has_group = False

    model_save_dir = f'lr={learning_rate}-'+ datetime.now().strftime('%y%m%d-%H%M%S') + f'CodeRefactorTesting'
    logdir = f'{root}/{model_save_dir}'
    ex.observers.append(FileStorageObserver.create(f'checkpoint/{model_save_dir}'))

def tensorboard_log(batch_visualize, model, valid_set, val_loader, train_set, ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer, reconstruction, has_features):
    scaling = HOP_LENGTH / SAMPLE_RATE
    # log various result from the validation audio
    model.eval()

    if ep%logging_freq==0 or ep==1:
        # on valid set
        with torch.no_grad():
            metrics, cm_dict_all = evaluate_prediction(valid_set, model, ep, scaling, has_state=has_features[0], has_group=has_features[1], eval_note=has_features[2], eval_tech=has_features[3])
            for key, values in metrics.items():
                if key.startswith('metric/'):
                    _, category, name = key.split('/')
                    # show metrics on terminal
                    print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                    if 'accuracy' in name or 'precision' in name or 'recall' in name or 'f1' in name:
                        writer.add_scalar(key, np.mean(values), global_step=ep)
        # for key, value in {**val_loss}.items():
        #     writer.add_scalar(key, value.item(), global_step=ep) 

        # visualized validation audio
        predictions, _, spec, post_a, post, latent = model.run_on_batch(batch_visualize)
        mel = spec[0]
        flux = spec[1]
        # Show the original transcription and spectrograms
        #loss = sum(losses.values())
        plot_post_and_latent(writer, ep, post_a, post, latent, flux)

        '''
        get transcriptions and confusion matrix
        '''
        transcriptions, cm_dict = get_transcription_and_cmx(batch_visualize['label'], predictions, ep, scaling)
        # plot features

        ########### Show the transcription result ###########
        print('Show the transcription result')
        plot_transcriptions(writer, ep, mel, transcriptions)

        ########### Plot confusion matrix ###########
        print('Plot confusion matrix')
        plot_confusion_matrices(writer, ep, cm_dict, cm_dict_all) 

        model.eval()
        test_losses = eval_model(model, ep, val_loader)
        for key, values in test_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)

def train_VAT_model(model, iteration, ep, l_loader, ul_loader, optimizer, scheduler, clip_gradient_norm, alpha):
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
        
        predictions, losses = model.run_on_batch(batch_l)

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
def train(spec, resume_iteration, sequence_length, w_size, n_heads, train_batch_size, val_batch_size, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, weight_decay, alpha, clip_gradient_norm, refresh, device, epoches, logdir, log, iteration, XI, eps, reconstruction, model_save_dir, has_note, has_tech, has_state, has_group, train_on): 
    print_config(ex.current_run)
    # flac for 16K audio
    has_features = (has_state, has_group, has_note, has_tech)
    train_set, valid_set = prepare_dataset(
        sequence_length=sequence_length,
        validation_length=sequence_length,
        refresh=refresh,
        device=device,
        audio_type='flac',
        dataset=train_on
    )  

    #generator=torch.Generator().manual_seed(42))
    
    # get weight of tech label for BCE loss
    bce_weights = compute_dataset_weight(device)

    print("train_set: ", len(train_set))
    print("valid_set: ", len(valid_set))
    # print("test_set: ", len(test_set))
    supervised_loader = DataLoader(train_set, train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_set, val_batch_size, shuffle=False, drop_last=True)
    batch_visualize = next(iter(val_loader)) # Getting one fixed batch for visualization   

    # model setting
    ds_ksize, ds_stride = 2, 2   
    model = UNet(ds_ksize,ds_stride, log=log, reconstruction=reconstruction,
                    mode=mode, spec=spec, device=device, XI=XI, eps=eps, weights=bce_weights)
    model.to(device)
    if resume_iteration is None:  
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)#,  weight_decay=weight_decay) #eps=1e-06
        resume_iteration = 0
    else: # Loading checkpoints and continue training
        model_path = os.path.join('checkpoint', f'{resume_iteration}.pt')
        model.load_state_dict(torch.load(model_path))
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join('checkpoint', 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    for ep in tqdm(range(1, epoches+1)):
       
        predictions, losses, train_losses, optimizer = train_VAT_model(model, iteration, ep, supervised_loader, None, optimizer, scheduler, clip_gradient_norm, alpha)            
        loss = sum(losses.values())

        # Logging results to tensorboard
        if ep == 1:
            writer = SummaryWriter(logdir) # create tensorboard logger     
     
        tensorboard_log(batch_visualize, model, valid_set, val_loader, train_set, ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer, reconstruction, has_features)        

        # Saving model
        if ep == 1 or (ep > 900 and (ep)%logging_freq == 0):
            torch.save(model.state_dict(), os.path.join('checkpoint', model_save_dir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join('checkpoint', 'last-optimizer-state.pt'))

        for key, values in train_losses.items():
            if key.startswith('loss/'):
                writer.add_scalar(key, np.mean(values), global_step=ep)
