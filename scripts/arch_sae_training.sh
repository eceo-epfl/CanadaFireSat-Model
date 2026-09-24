#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1.

max_epochs=60
experiment="resamp-long-delta-keyMSCLIP25k-init-W-high-auxk-topk-20-2048-msclip-10x-3gram"
nb_concepts=2048
nb_k=20
model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
sae="topk_arch"
# use_archetypical=True

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} sae=${sae} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"