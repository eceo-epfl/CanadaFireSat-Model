#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

max_epochs=20
experiment="veryhigh-auxk-topk-40-4096-msclip-10x-ngram"
nb_concepts=4096
nb_k=40
model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
sae="topk_arch"
use_archetypical=True

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} sae=${sae} use_archetypical=${use_archetypical} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"
