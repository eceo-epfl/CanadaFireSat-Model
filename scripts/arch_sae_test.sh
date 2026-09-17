#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

max_epochs=20
experiment="test-topk-40-4096-msclip-10x"
nb_concepts=4096
nb_k=40
model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
test_only=True
sae_ckpt_path="./results/logs/train-sae/runs/2026-08-24_13-00-56/checkpoints/epoch_019.ckpt"

python experiments/concept_bottleneck/sae/train_sae.py sae_max_epochs=${max_epochs} \
nb_concepts=${nb_concepts} nb_k=${nb_k} model_ckpt=${model_ckpt} test_only=${test_only} sae_ckpt_path=${sae_ckpt_path} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"
