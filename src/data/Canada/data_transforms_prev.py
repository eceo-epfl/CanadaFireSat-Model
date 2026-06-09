"""Functions to build the different transformation pipelines"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torchvision import transforms

from deepsat.data.PASTIS24.data_transforms import TileDates, UnkMask
from src.constants import (
    BANDS_ALL,
    ENV_SOURCE_COLS,
    LOW_SOURCE,
    MID_SOURCE,
    TAB_SOURCE_COLS,
)
from src.data.augmentations import (
    Concat,
    Crop,
    CutOrPad,
    DownSampleLab,
    EnvConcat,
    EnvNormalize,
    EnvRescale,
    EnvTileDates,
    EnvToTensor,
    EnvToTHWC,
    GaussianNoise,
    HVFlip,
    Normalize,
    Rescale,
    ResizedCrop,
    TabNormalize,
    TabTileDates,
    TabToTensor,
    TileLocs,
    ToTensor,
    ToTHWC,
)
from src.data.utils import _extract_stats


# Test Extension with ColorJittering (RGB only), Coarse Dropout (Extend Masking), Mixup (Potentially in Dataset)
def Canada_segmentation_transform(
    model_config: Dict[str, Any],
    mean_file: os.PathLike,
    std_file: os.PathLike,
    is_training: bool,
    is_eval: bool = False,
    bands: List[str] = BANDS_ALL,
    with_doy: bool = True,
    with_loc: bool = True,
    img_only: bool = True,
    **kwargs,
) -> transforms.Compose:
    """SITS augmentation pipeline

    Args:
        model_config (Dict[str, Any]): Config of the Model
        mean_file (os.PathLike): File containing the mean of all bands
        std_file (os.PathLike): File containing the std of all bands
        is_training (bool): Flag if the pipeline is done for training
        is_eval (bool, optional): Flag if the pipeline is done for final evaluation. Defaults to False.
        bands (List[str], optional): Lists of Bands to include. Defaults to BANDS_ALL.
        with_doy (bool, optional): Flag if we use day of the year. Defaults to True.
        with_loc (bool, optional): Flag if we use the localization. Defaults to True.
        img_only (bool, optional): Flag if we use SITS only. Defaults to True.

    Returns:
        transforms.Compose: Output tranformation pipeline
    """

    # Extract the mean and std from the files
    mean_array = _extract_stats(mean_file, bands)
    std_array = _extract_stats(std_file, bands)

    # Custom Bands Transforms
    band_transform_list = [
        ToTensor(with_loc=with_loc),
        Rescale(output_size=(model_config["input_img_res"], model_config["input_img_res"])),
        Concat(concat_keys=["10x", "20x", "60x"]),  # Order is important for RGB weights
        Normalize(mean=mean_array, std=std_array),
    ]

    # Regularization Image Transforms
    if is_training:
        img_transform_list = [
            Crop(
                img_size=model_config["input_img_res"],
                crop_size=model_config["img_res"],
                random=True,
                ground_truths=["labels"],
                with_loc=with_loc,
            ),
            ResizedCrop(
                out_size=model_config["img_res"],
                scale=(0.9, 1.0),
                prob=0.5,
                ground_truths=["labels"],
                with_loc=with_loc,
            ),
            DownSampleLab(out_H=model_config["out_H"], out_W=model_config["out_W"]),
            HVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=with_loc) if img_only else transforms.Lambda(lambda x: x),
            GaussianNoise(var_limit=(0.01, 0.1), p=0.5),
        ]

        if with_loc:
            img_transform_list.append(TileLocs())

        if with_doy:
            img_transform_list.append(TileDates(H=model_config["img_res"], W=model_config["img_res"], doy_bins=None))

        img_transform_list.append(CutOrPad(max_seq_len=model_config["train_max_seq_len"], sampling_type="random"))
        img_transform_list.append(UnkMask(unk_class=-999, ground_truth_target="labels"))
        img_transform_list.append(ToTHWC())

    else:
        img_transform_list = [
            Crop(
                img_size=model_config["input_img_res"],
                crop_size=model_config["img_res"],
                random=False,
                ground_truths=["labels"],
                with_loc=with_loc,
            ),
            DownSampleLab(out_H=model_config["out_H"], out_W=model_config["out_W"]),
        ]

        if with_loc:
            img_transform_list.append(TileLocs())

        if with_doy:
            img_transform_list.append(TileDates(H=model_config["img_res"], W=model_config["img_res"], doy_bins=None))

        if not is_eval:
            img_transform_list.append(
                CutOrPad(
                    max_seq_len=model_config["val_max_seq_len"],
                    sampling_type=kwargs["eval_sampling"] if "eval_sampling" in kwargs else "start",
                )
            )

        elif is_eval and "test_max_seq_len" in model_config:
            img_transform_list.append(
                CutOrPad(
                    max_seq_len=model_config["test_max_seq_len"],
                    sampling_type=kwargs["eval_sampling"] if "eval_sampling" in kwargs else "start",
                )
            )

        img_transform_list.append(UnkMask(unk_class=-999, ground_truth_target="labels"))
        img_transform_list.append(ToTHWC())

    total_transform_list = band_transform_list + img_transform_list

    return transforms.Compose(total_transform_list)



def TabCanada_segmentation_transform(
    model_config: Dict[str, Any],
    stats_dir: os.PathLike,
    tab_source_cols: Dict[str, List[str]] = TAB_SOURCE_COLS,
    with_doy: bool = True,
    is_training=True,
    **kwargs,
) -> transforms.Compose:
    """Tabular data augmentation pipeline

    Args:
        model_config (Dict[str, Any]): Config of the Model
        stats_dir (os.PathLike): Directory containing mean and std files
        tab_source_cols (Dict[str, List[str]], optional): Dictionary mapping sources to target variables. Defaults to TAB_SOURCE_COLS.
        with_doy (bool, optional): Flag if we use day of the year. Defaults to True.
        is_training (bool, optional): Flag if the pipeline is done for training. Defaults to True.

    Returns:
        transforms.Compose: Output tabular pipeline
    """

    tot_mean_ls = []
    tot_std_ls = []

    for source, cols in tab_source_cols.items():
        with open(Path(stats_dir) / f"{source}_mean.json", "r") as f:
            json_mean = json.load(f)

        cols = sorted(cols)  # Ensure the order is the same as in the model

        mean_array = np.array([json_mean[col] for col in cols]).reshape(1, len(cols)).astype(np.float32)  # Tab: N*C*S
        tot_mean_ls.append(mean_array)

        with open(Path(stats_dir) / f"{source}_std.json", "r") as f:
            json_std = json.load(f)

        std_array = np.array([json_std[col] for col in cols]).reshape(1, len(cols)).astype(np.float32)
        tot_std_ls.append(std_array)

    tot_mean_array = np.concatenate(tot_mean_ls, axis=1)
    tot_std_array = np.concatenate(tot_std_ls, axis=1)

    transform_list = []
    transform_list.append(TabToTensor())  # data from numpy arrays to torch.float32
    transform_list.append(TabNormalize(mean=tot_mean_array, std=tot_std_array))  # normalize all inputs individually

    if with_doy:
        transform_list.append(TabTileDates())  # tile day and year to shape Tx1

    if is_training and "tab_train_max_seq_len" in model_config:
        transform_list.append(
            CutOrPad(max_seq_len=model_config["tab_train_max_seq_len"], sampling_type="random", mode="tab")
        )
    elif "tab_val_max_seq_len" in model_config:
        transform_list.append(
            CutOrPad(max_seq_len=model_config["tab_val_max_seq_len"], sampling_type="start", mode="tab")
        )

    return transforms.Compose(transform_list)


def EnvCanada_segmentation_transform(
    model_config: Dict[str, Any],
    stats_dir: os.PathLike,
    tab_source_cols: Dict[str, List[str]] = ENV_SOURCE_COLS,
    with_doy: bool = True,
    with_loc: bool = True,
    env_only: bool = False,
    is_training: bool = True,
    **kwargs,
) -> transforms.Compose:
    """Environmental tiles data augmentation pipeline

    Args:
        model_config (Dict[str, Any]): Config of the Model
        stats_dir (os.PathLike): Directory containing mean and std files
        tab_source_cols (Dict[str, List[str]], optional): Dictionary mapping soources to target variables. Defaults to ENV_SOURCE_COLS.
        with_doy (bool, optional): Flag if we use day of the year. Defaults to True.
        with_loc (bool, optional): Flag if we use the localization. Defaults to True.
        env_only (bool, optional): Flag if the pipeline is done for environmental data only. Defaults to False.
        is_training (bool, optional): Flag if the pipeline is done for training. Defaults to True.

    Returns:
        transforms.Compose: Output environmental pipeline
    """

    tot_mean_ls = {}
    tot_std_ls = {}

    for source, cols in tab_source_cols.items():
        with open(Path(stats_dir) / f"{source}_mean.json", "r") as f:
            json_mean = json.load(f)

        cols = sorted(cols)  # Ensure the order is the same as in the model

        mean_array = (
            np.array([json_mean[col] for col in cols]).reshape(1, len(cols), 1, 1).astype(np.float32)
        )  # Env: N*C*H*W
        tot_mean_ls[source] = mean_array

        with open(Path(stats_dir) / f"{source}_std.json", "r") as f:
            json_std = json.load(f)

        std_array = np.array([json_std[col] for col in cols]).reshape(1, len(cols), 1, 1).astype(np.float32)
        tot_std_ls[source] = std_array

    mid_mean_array = np.concatenate([tot_mean_ls[source] for source in MID_SOURCE], axis=1)
    low_mean_array = np.concatenate([tot_mean_ls[source] for source in LOW_SOURCE], axis=1)

    mid_std_array = np.concatenate([tot_std_ls[source] for source in MID_SOURCE], axis=1)
    low_std_array = np.concatenate([tot_std_ls[source] for source in LOW_SOURCE], axis=1)

    transform_list = []
    transform_list.append(EnvToTensor(with_loc=with_loc))  # data from numpy arrays to torch.float32
    transform_list.append(EnvRescale(mid_size=model_config["mid_input_res"], low_size=model_config["low_input_res"]))
    transform_list.append(EnvConcat(mid_keys=MID_SOURCE, low_keys=LOW_SOURCE))  # concat mid and low sources
    transform_list.append(
        EnvNormalize(mid_mean=mid_mean_array, mid_std=mid_std_array, low_mean=low_mean_array, low_std=low_std_array)
    )  # normalize all inputs individually
    transform_list.append(
        DownSampleLab(out_H=model_config["out_H"], out_W=model_config["out_W"])
        if env_only
        else transforms.Lambda(lambda x: x)
    )

    if is_training:
        transform_list.extend(
            [
                (
                    HVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=with_loc)
                    if env_only
                    else transforms.Lambda(lambda x: x)
                ),
                GaussianNoise(var_limit=(0.01, 0.1), p=0.5),
            ]
        )
        if with_doy:
            transform_list.append(
                EnvTileDates(
                    mid_H=model_config["mid_input_res"],
                    mid_W=model_config["mid_input_res"],
                    low_H=model_config["low_input_res"],
                    low_W=model_config["low_input_res"],
                    doy_bins=None,
                )
            )  # tile day and year to shape Tx1
        if "env_train_max_seq_len" in model_config:
            transform_list.append(
                CutOrPad(max_seq_len=model_config["env_train_max_seq_len"], sampling_type="random", mode="env")
            )

    elif "env_val_max_seq_len" in model_config:
        if with_doy:
            transform_list.append(
                EnvTileDates(
                    mid_H=model_config["mid_input_res"],
                    mid_W=model_config["mid_input_res"],
                    low_H=model_config["low_input_res"],
                    low_W=model_config["low_input_res"],
                    doy_bins=None,
                )
            )  # tile day and year to shape Tx1
        transform_list.append(
            CutOrPad(max_seq_len=model_config["env_val_max_seq_len"], sampling_type="start", mode="env")
        )
    elif with_doy:
        transform_list.append(
            EnvTileDates(
                mid_H=model_config["mid_input_res"],
                mid_W=model_config["mid_input_res"],
                low_H=model_config["low_input_res"],
                low_W=model_config["low_input_res"],
                doy_bins=None,
            )
        )  # tile day and year to shape Tx1

    if with_loc:
        raise NotImplementedError("Location information is not yet implemented for the Environment Canada data.")

    transform_list.append(EnvToTHWC())

    return transforms.Compose(transform_list)
