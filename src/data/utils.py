"""Utils functions for Data Processing"""
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms

from src.constants import BANDS_10, BANDS_20, BANDS_60
from src.data.augmentations import CutOrPad


# Adapted from deepsat.data
def segmentation_ground_truths(sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = sample["labels"]
    if "unk_masks" in sample.keys():
        unk_masks = sample["unk_masks"]
    else:
        unk_masks = None

    if "edge_labels" in sample.keys():
        edge_labels = sample["edge_labels"]
        return labels, edge_labels, unk_masks
    return labels, unk_masks


# Keep the relative order of BANDS_10, BANDS_20, BANDS_60 for normalization: based on preprocessing and/or __adapt__ method
def _extract_stats(stats_path: os.PathLike, bands: List[str]) -> np.ndarray:

    with open(stats_path, "r") as f:
        json_stats = json.load(f)

    stats_array_10x = (
        np.array([json_stats[band] for band in BANDS_10 if band in bands]).reshape(1, -1, 1, 1).astype(np.float32)
    )  # To be T * C * H * W
    stats_array_20x = (
        np.array([json_stats[band] for band in BANDS_20 if band in bands]).reshape(1, -1, 1, 1).astype(np.float32)
    )  # To be T * C * H * W
    stats_array_60x = (
        np.array([json_stats[band] for band in BANDS_60 if band in bands]).reshape(1, -1, 1, 1).astype(np.float32)
    )  # To be T * C * H * W

    stats_array = np.concatenate([stats_array_10x, stats_array_20x, stats_array_60x], axis=1)
    return stats_array


def _load_json_stats(stats_dir: Path, source: str, cols: List[str], shape: tuple):
    cols = sorted(cols)
    with open(stats_dir / f"{source}_mean.json") as f:
        mean = np.array([json.load(f)[c] for c in cols], dtype=np.float32).reshape(shape)
    with open(stats_dir / f"{source}_std.json") as f:
        std = np.array([json.load(f)[c] for c in cols], dtype=np.float32).reshape(shape)
    return mean, std


def _cutorpad(max_seq_len: Optional[int], sampling_type: str, mode: str = "image", flag_doy_process: bool = False):
    if max_seq_len is None:
        if mode == "image":
            return [transforms.Lambda(
                lambda sample: {**sample, "seq_lengths": deepcopy(sample["inputs"].shape[0])}
            )]
        return []
    return [CutOrPad(max_seq_len=max_seq_len, sampling_type=sampling_type, mode=mode, flag_doy_process=flag_doy_process)]
