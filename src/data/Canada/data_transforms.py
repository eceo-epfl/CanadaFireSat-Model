"""Transform pipeline registry — one config dict, one build call."""
from functools import partial
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torchvision import transforms

from deepsat.data.PASTIS24.data_transforms import TileDates, UnkMask
from src.constants import BANDS_ALL, ENV_SOURCE_COLS, LOW_SOURCE, MID_SOURCE, MSCLIP_ORDER_10, S2_UINT_TO_REFLECTANCE, TAB_SOURCE_COLS
from src.data.augmentations import (
    Concat, Crop, CutOrPad, DownSampleLab, EnvConcat, EnvNormalize,
    EnvRescale, EnvTileDates, EnvToTensor, EnvToTHWC, GaussianNoise,
    HVFlip, Normalize, ProcessDoy, ReorderBands, Rescale, ResizedCrop, TabNormalize, TabTileDates,
    TabToTensor, TileLocs, ToTensor, ToTHWC,
)
from src.data.utils import _extract_stats, _load_json_stats, _cutorpad


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PIPELINE_REGISTRY: Dict[str, type] = {}

def register_pipeline(name: str):
    """Decorator to register a pipeline class under a given name."""
    def decorator(cls):
        if name in _PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline '{name}' is already registered.")
        _PIPELINE_REGISTRY[name] = cls
        return cls
    return decorator

def build_pipeline(name: str, config: Dict[str, Any], is_training: bool, is_eval: bool = False):
    """Entry point: look up pipeline by name, instantiate with config, build."""
    if name not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline '{name}'. "
            f"Available: {list(_PIPELINE_REGISTRY.keys())}"
        )
    return _PIPELINE_REGISTRY[name](config).build(is_training, is_eval)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TransformPipeline:
    """
    Base class for all pipelines. Subclasses receive the raw config dict
    directly — no separate dataclass needed. Each subclass declares its
    REQUIRED_KEYS and OPTIONAL_KEYS so misconfiguration is caught early.

    Config keys are just plain strings, e.g.:
        config = {
            "model_config": {...},
            "mean_file": "path/to/mean.npy",
            "bands": ["B02", "B03", ...],
        }
    """

    REQUIRED_KEYS: List[str] = []
    OPTIONAL_KEYS: Dict[str, Any] = {}  # key -> default value

    def __init__(self, config: Dict[str, Any]):
        self._validate(config)
        # Merge defaults for optional keys
        self.config = {**self.OPTIONAL_KEYS, **config}

    def _validate(self, config: Dict[str, Any]):
        missing = [k for k in self.REQUIRED_KEYS if k not in config]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required config keys: {missing}")

    def base_transforms(self) -> List:
        return []

    def train_transforms(self) -> List:
        return []

    def eval_transforms(self, is_eval: bool = False) -> List:
        return []

    def build(self, is_training: bool, is_eval: bool = False) -> transforms.Compose:
        split_transforms = self.train_transforms() if is_training else self.eval_transforms(is_eval)
        return transforms.Compose(self.base_transforms() + split_transforms)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

@register_pipeline("sits")
class SITSTransformPipeline(TransformPipeline):

    REQUIRED_KEYS = ["model_config", "mean_file", "std_file"]
    OPTIONAL_KEYS = {
        "bands": BANDS_ALL,
        "with_doy": True,
        "with_loc": True,
        "img_only": True,
        "eval_sampling": "start",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        c = self.config
        self.mean = _extract_stats(c["mean_file"], c["bands"])
        self.std  = _extract_stats(c["std_file"],  c["bands"])
        mc = c["model_config"]
        self._img_res       = mc["img_res"]
        self._input_img_res = mc["input_img_res"]
        self._out_H         = mc["out_H"]
        self._out_W         = mc["out_W"]
        self._train_seq_len = mc.get("train_max_seq_len")
        self._val_seq_len   = mc.get("val_max_seq_len")
        self._test_seq_len  = mc.get("test_max_seq_len")

    def base_transforms(self):
        c = self.config
        return [
            ToTensor(with_loc=c["with_loc"]),
            Rescale(output_size=(self._input_img_res, self._input_img_res)),
            Concat(concat_keys=["10x", "20x", "60x"]),
            Normalize(mean=self.mean, std=self.std),
        ]

    def train_transforms(self):
        c = self.config
        t = [
            Crop(img_size=self._input_img_res, crop_size=self._img_res,
                 random=True, ground_truths=["labels"], with_loc=c["with_loc"]),
            ResizedCrop(out_size=self._img_res, scale=(0.9, 1.0), prob=0.5,
                        ground_truths=["labels"], with_loc=c["with_loc"]),
            DownSampleLab(out_H=self._out_H, out_W=self._out_W),
            HVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=c["with_loc"])
            if c["img_only"] else transforms.Lambda(lambda x: x),
            GaussianNoise(var_limit=(0.01, 0.1), p=0.5),
        ]
        if c["with_loc"]: t.append(TileLocs())
        if c["with_doy"]: t.append(TileDates(H=self._img_res, W=self._img_res, doy_bins=None))
        t += _cutorpad(self._train_seq_len, "random")
        t += [UnkMask(unk_class=-999, ground_truth_target="labels"), ToTHWC()]
        return t

    def eval_transforms(self, is_eval: bool = False):
        c = self.config
        t = [
            Crop(img_size=self._input_img_res, crop_size=self._img_res,
                 random=False, ground_truths=["labels"], with_loc=c["with_loc"]),
            DownSampleLab(out_H=self._out_H, out_W=self._out_W),
        ]
        if c["with_loc"]: t.append(TileLocs())
        if c["with_doy"]: t.append(TileDates(H=self._img_res, W=self._img_res, doy_bins=None))
        seq_len = self._test_seq_len if is_eval else self._val_seq_len
        t += _cutorpad(seq_len, c["eval_sampling"])
        t += [UnkMask(unk_class=-999, ground_truth_target="labels"), ToTHWC()]
        return t


"""
@register_pipeline("tab")
class TabTransformPipeline(TransformPipeline):

    REQUIRED_KEYS = ["model_config", "stats_dir"]
    OPTIONAL_KEYS = {
        "tab_source_cols": TAB_SOURCE_COLS,
        "with_doy": True,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        c = self.config
        mc = c["model_config"]
        self._train_seq_len = mc.get("tab_train_max_seq_len")
        self._val_seq_len   = mc.get("tab_val_max_seq_len")

        means, stds = [], []
        for source, cols in c["tab_source_cols"].items():
            m, s = _load_json_stats(Path(c["stats_dir"]), source, cols, shape=(1, len(cols)))
            means.append(m); stds.append(s)
        self.mean = np.concatenate(means, axis=1)
        self.std  = np.concatenate(stds,  axis=1)

    def base_transforms(self):
        t = [TabToTensor(), TabNormalize(mean=self.mean, std=self.std)]
        if self.config["with_doy"]:
            t.append(TabTileDates())
        return t

    def train_transforms(self):
        return _cutorpad(self._train_seq_len, "random", mode="tab")

    def eval_transforms(self):
        return _cutorpad(self._val_seq_len, "start", mode="tab")


@register_pipeline("env")
class EnvTransformPipeline(TransformPipeline):

    REQUIRED_KEYS = ["model_config", "stats_dir"]
    OPTIONAL_KEYS = {
        "tab_source_cols": ENV_SOURCE_COLS,
        "with_doy": True,
        "with_loc": False,
        "env_only": False,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        c = self.config
        if c["with_loc"]:
            raise NotImplementedError("Location information is not yet implemented for Environment Canada data.")

        mc = c["model_config"]
        self._mid_res       = mc["mid_input_res"]
        self._low_res       = mc["low_input_res"]
        self._out_H         = mc["out_H"]
        self._out_W         = mc["out_W"]
        self._train_seq_len = mc.get("env_train_max_seq_len")
        self._val_seq_len   = mc.get("env_val_max_seq_len")

        mid_means, mid_stds, low_means, low_stds = [], [], [], []
        for source, cols in c["tab_source_cols"].items():
            m, s = _load_json_stats(Path(c["stats_dir"]), source, cols, shape=(1, len(cols), 1, 1))
            (mid_means if source in MID_SOURCE else low_means).append(m)
            (mid_stds  if source in MID_SOURCE else low_stds ).append(s)

        self.mid_mean = np.concatenate(mid_means, axis=1)
        self.mid_std  = np.concatenate(mid_stds,  axis=1)
        self.low_mean = np.concatenate(low_means, axis=1)
        self.low_std  = np.concatenate(low_stds,  axis=1)

    def _doy_transform(self):
        return EnvTileDates(
            mid_H=self._mid_res, mid_W=self._mid_res,
            low_H=self._low_res, low_W=self._low_res,
            doy_bins=None,
        )

    def base_transforms(self):
        c = self.config
        return [
            EnvToTensor(with_loc=c["with_loc"]),
            EnvRescale(mid_size=self._mid_res, low_size=self._low_res),
            EnvConcat(mid_keys=MID_SOURCE, low_keys=LOW_SOURCE),
            EnvNormalize(mid_mean=self.mid_mean, mid_std=self.mid_std,
                         low_mean=self.low_mean, low_std=self.low_std),
            DownSampleLab(out_H=self._out_H, out_W=self._out_W)
            if c["env_only"] else transforms.Lambda(lambda x: x),
        ]

    def train_transforms(self):
        c = self.config
        t = [
            HVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=c["with_loc"])
            if c["env_only"] else transforms.Lambda(lambda x: x),
            GaussianNoise(var_limit=(0.01, 0.1), p=0.5),
        ]
        if c["with_doy"]: t.append(self._doy_transform())
        t += _cutorpad(self._train_seq_len, "random", mode="env")
        t.append(EnvToTHWC())
        return t

    def eval_transforms(self):
        t = []
        if self.config["with_doy"]: t.append(self._doy_transform())
        t += _cutorpad(self._val_seq_len, "start", mode="env")
        t.append(EnvToTHWC())
        return t
"""

@register_pipeline("msclip")
class MSCLIPTransformPipeline(TransformPipeline):

    REQUIRED_KEYS = ["model_config", "mean_file", "std_file"]
    OPTIONAL_KEYS = {
        "bands": BANDS_ALL,
        "with_doy": True,
        "with_loc": True,
        "img_only": True,
        "eval_sampling": "start",
        "use_msclip_norm": True,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        c = self.config
        self.mean = _extract_stats(c["mean_file"], c["bands"])
        self.std  = _extract_stats(c["std_file"],  c["bands"])
        mc = c["model_config"]
        self._img_res       = mc["img_res"]
        self._input_img_res = mc["input_img_res"]
        self._out_H         = mc["out_H"]
        self._out_W         = mc["out_W"]
        self._train_seq_len = mc.get("train_max_seq_len")
        self._val_seq_len   = mc.get("val_max_seq_len")
        self._test_seq_len  = mc.get("test_max_seq_len")

    @staticmethod
    def _multiply_inputs(sample, factor: float):
        # sample is a dict; copy if you want to be safe
        sample = dict(sample)
        sample["inputs"] = sample["inputs"] * factor
        return sample

    def base_transforms(self):
        c = self.config
        t =  [
            ToTensor(with_loc=c["with_loc"]),
            Rescale(output_size=(self._input_img_res, self._input_img_res)),
            Concat(concat_keys=["10x", "20x", "60x"]),
            ReorderBands(order=MSCLIP_ORDER_10),
        ]
        if c["use_msclip_norm"]: t.append(transforms.Lambda(partial(self._multiply_inputs, factor=S2_UINT_TO_REFLECTANCE)))

        t.append(Normalize(mean=self.mean, std=self.std))
        return t

    def train_transforms(self):
        c = self.config
        t = [
            Crop(img_size=self._input_img_res, crop_size=self._img_res,
                 random=True, ground_truths=["labels"], with_loc=c["with_loc"]),
            ResizedCrop(out_size=self._img_res, scale=(0.9, 1.0), prob=0.5,
                        ground_truths=["labels"], with_loc=c["with_loc"]),
            DownSampleLab(out_H=self._out_H, out_W=self._out_W),
            HVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=c["with_loc"])
            if c["img_only"] else transforms.Lambda(lambda x: x), # Do we need this ?
            # GaussianNoise(var_limit=(0.01, 0.1), p=0.5),
        ]
        if c["with_loc"]: t.append(TileLocs())
        if c["with_doy"]: t.append(ProcessDoy(H=self._img_res, W=self._img_res, doy_bins=None, max_seq_len=None))
        t += _cutorpad(self._train_seq_len, "random", flag_doy_process=c["with_doy"])
        t += [UnkMask(unk_class=-999, ground_truth_target="labels")]
        return t

    def eval_transforms(self, is_eval: bool = False):
        c = self.config
        t = [
            Crop(img_size=self._input_img_res, crop_size=self._img_res,
                 random=False, ground_truths=["labels"], with_loc=c["with_loc"]),
            DownSampleLab(out_H=self._out_H, out_W=self._out_W),
        ]
        if c["with_loc"]: t.append(TileLocs())
        if c["with_doy"]: t.append(ProcessDoy(H=self._img_res, W=self._img_res, doy_bins=None, max_seq_len=None))
        seq_len = self._test_seq_len if is_eval else self._val_seq_len
        t += _cutorpad(seq_len, c["eval_sampling"], flag_doy_process=c["with_doy"])
        t += [UnkMask(unk_class=-999, ground_truth_target="labels")]
        return t