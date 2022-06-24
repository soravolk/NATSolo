import os
from yacs.config import CfgNode as CN

config = CN()

config.root = 'run'
config.ckpt_root = 'ckpt'
config.device = 'cpu'
config.log = True
config.resume_iteration = True

config.dataset = CN()
config.dataset.name = 'PanoCorBonDataset'
config.dataset.common_kwargs = CN()
config.dataset.common_kwargs.sequence_length = 327680 
config.dataset.train_kwargs = CN(new_allowed=True)
config.dataset.valid_kwargs = CN(new_allowed=True)

config.training = CN()
config.training.epoches = 2000
config.training.iteration = 10 
config.training.batch_size = 8 
config.training.learning_rate = 1e-3
config.training.learning_rate_decay_steps = 1000
config.training.learning_rate_decay_rate = 0.98
config.training.clip_gradient_norm = 3
config.training.refresh= False

config.val = CN(new_allowed=True)
config.val.batch_size = 3

config.model = CN()
config.model.file = 'model.UNet'
config.model.modelclass = 'UNet'
config.model.VAT = True
config.model.reconstruction = True
config.model.kwargs = CN(new_allowed=True)
config.model.kwargs.ds_ksize = (2,2)
config.model.kwargs.ds_kernel =(2,2)
config.model.kwargs.mode = 'imagewise'
config.model.kwargs.sparsity = 2
config.model.kwargs.output_channel = 2
config.model.kwargs.logging_freq = 100
config.model.kwargs.saving_freq = 200
config.model.kwargs.XI = 1e-6
config.model.kwargs.VAT_start = 0
config.model.kwargs.n_heads = 4
config.model.kwargs.alpha = 1 
config.model.kwargs.w_size = 31
config.model.kwargs.eps = 1.3


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

def infer_exp_id(cfg_path):
    cfg_path = cfg_path.split('config/')[-1]
    if cfg_path.endswith('.yaml'):
        cfg_path = cfg_path[:-len('.yaml')]
    return '_'.join(cfg_path.split('/'))