#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1.

max_epochs=20
experiment="TOY-10Xlr-vpt-1e-4-l1-warmup-cosine-probe-training"
model_ckpt="MIXLOSS_HIGHLR10X_VPT_SITS_ONLY/MSClipFacto-18-f1-0.48.ckpt"
sae_folder_path="2026-10-14_08-40-46"
sae_epoch_num="epoch_017"
lr_base=5e-5
lr_min=1e-6
lr_start=1e-6

python experiments/concept_bottleneck/bottleneck/train_probe.py --config-name train_probe_toy.yaml \
probe_max_epochs=${max_epochs} \
probe.sae_folder_path=${sae_folder_path} \
probe.sae_epoch_num=${sae_epoch_num} \
model_ckpt=${model_ckpt} \
probe.scheduler_kwargs.lr_base=${lr_base} \
probe.scheduler_kwargs.lr_min=${lr_min} \
probe.scheduler_kwargs.lr_start=${lr_start} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"