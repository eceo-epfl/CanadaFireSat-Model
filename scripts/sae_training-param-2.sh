#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

max_epochs=20
experiment="baseline-topk-20-2046-auxk-0.5-lr-0.0005-mix"
nb_concepts=2048
nb_k=20
#model_ckpt="MIXLOSS_HIGHLR10X_VPT_SITS_ONLY/MSClipFacto-18-f1-0.48.ckpt"
model_ckpt="STRIP_MIXLOSS_VPT_CLEARCLIP_HIGH10X_SITS_ONLY/MSClipFacto-16-f1-0.48.ckpt"
sae_lr=0.0005
penalty=0.5

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} \
sae_lr=${sae_lr} sae.criterion_kwargs.penalty=${penalty} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"
