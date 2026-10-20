#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

max_epochs=20
experiment="baseline-apertus-mid-auxk-topk-10-1024-msclip-10x-3gram"
nb_concepts=1024
nb_k=10
#model_ckpt="MIXLOSS_HIGHLR10X_VPT_SITS_ONLY/MSClipFacto-18-f1-0.48.ckpt"
model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
sae_lr=0.0005
penalty=0.01

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"
