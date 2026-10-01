#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1.

max_epochs=20
experiment="debug-probe-training"
model_ckpt="MIXLOSS_HIGHLR10X_MSCLIP_SITS_ONLY/MSClipFacto-17-f1-0.46.ckpt"
sae_folder_path="2026-09-24_14-20-52"
sae_epoch_num="epoch_052"

python experiments/concept_bottleneck/bottleneck/train_probe.py probe_max_epochs=${max_epochs} \
probe.sae_folder_path=${sae_folder_path} \
probe.sae_epoch_num=${sae_epoch_num} \
model_ckpt=${model_ckpt} \
logger=wandb logger.wandb.name="${experiment}_$(date +%Y%m%d-%H%M%S)"