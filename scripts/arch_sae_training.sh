#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1.

max_epochs=20
experiment="resample-delta-1.-apertus-init-W-auxk-topk-16-1024-vpt-10x-3gram"
nb_concepts=1024
nb_k=16
model_ckpt="VPTFIX_MIXLOSS_VPT_HIGH10X_SITS_ONLY/MSClipFacto-18-f1-0.48.ckpt"
#model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
sae="topk_arch"
sae_lr=0.0005
penalty=0.01
resample_every_n_epochs=1
delta=1.
l2_norm=True # Text archetypes are unit-norm
# use_archetypical=True

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} sae=${sae} \
sae_lr=${sae_lr} sae.criterion_kwargs.penalty=${penalty} \
sae.resample_every_n_epochs=${resample_every_n_epochs} sae.arch_kwargs.delta=${delta} sae_l2_norm=${l2_norm} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"