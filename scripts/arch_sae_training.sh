#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1.

max_epochs=20
experiment="resample-delta-1.01-apertus-init-W-high-auxk-topk-20-2048-msclip-10x-3gram"
nb_concepts=2048
nb_k=20
#model_ckpt="MIXLOSS_HIGHLR10X_VPT_SITS_ONLY/MSClipFacto-18-f1-0.48.ckpt"
model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
sae="topk_arch"
sae_lr=0.0005
penalty=0.1
resample_every_n_epochs=1
delta=1.01
# use_archetypical=True

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} sae=${sae} \
sae_lr=${sae_lr} sae.criterion_kwargs.penalty=${penalty} \
sae.resample_every_n_epochs=${resample_every_n_epochs} sae.arch_kwargs.delta=${delta} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"