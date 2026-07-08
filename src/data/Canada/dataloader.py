"""Build Dataloader for SITS from *.npy files"""

from __future__ import division, print_function

import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.Canada.sampler import FileIDSampler, FWISampler

warnings.filterwarnings("ignore")

from src.constants import (
    BANDS_10,
    BANDS_20,
    BANDS_60,
    BANDS_ALL,
    CANADA_REF_COLS,
    LABEL_IMG_RES,
    TEST_FILTER,
    TRAIN_FILTER,
    VAL_FILTER,
)


def get_dataloader(
    paths_file: Union[str, List[str]],
    root_dir: Union[str, List[str]],
    label_dir: str,
    transform: transforms.Compose = None,
    split: str = "train",
    bands: List[str] = BANDS_ALL,
    suffix: str = "one",
    fast: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    return_paths: bool = False,
    my_collate: Callable = None,
    target_file_id: Optional[int] = None,
    fwi_th: float = None,
    with_loc: bool = False,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Get Dataloader for SITS only.

    Args:
        paths_file (Union[str, List[str]]): Path of the file(s) listing the samples
        root_dir (Union[str, List[str]]): Root directories of the SITS
        label_dir (str): Directory of the positive samples: binary labels
        transform (transforms.Compose, optional): Data augmentation pipeline. Defaults to None.
        split (str, optional): String indicating the split. Defaults to "train".
        bands (List[str], optional): List of bands to use. Defaults to BANDS_ALL.
        suffix (str, optional): Suffix to access the right npy files. Defaults to "one".
        fast (bool, optional): Flag to access fast loading. Defaults to True.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for loading samples. Defaults to 4.
        shuffle (bool, optional): Flag to shuffle samples. Defaults to True.
        return_paths (bool, optional): Flag to return the sample info. Defaults to False.
        my_collate (Callable, optional): Collate function for the batch. Defaults to None.
        target_file_id (Optional[int], optional): File Id to sample only one specific sample type. Defaults to None.
        fwi_th (float, optional): FWI threshild for sampling specific fire risk. Defaults to None.
        with_loc (bool, optional): Flag to use localization. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: SITS Dataloader
    """

    dataset = SatImDataset(
        json_file=paths_file,
        root_dir=root_dir,
        label_dir=label_dir,
        transform=transform,
        split=split,
        return_paths=return_paths,
        bands=bands,
        suffix=suffix,
        fast=fast,
        with_loc=with_loc,
    )

    if target_file_id is not None:
        sampler = FileIDSampler(dataset, target_file_id, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=my_collate,
            sampler=sampler,
            prefetch_factor=2,
        )

    elif fwi_th is not None:
        sampler = FWISampler(dataset, fwi_th, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=my_collate,
            sampler=sampler,
            prefetch_factor=2,
        )

    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=my_collate,
            prefetch_factor=2,
        )

    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(
        self,
        json_file: Union[str, List[str]],
        root_dir: Union[str, List[str]],
        label_dir: str,
        transform: transforms.Compose = None,
        split: str = "train",
        return_paths: bool = False,
        bands: List[str] = BANDS_ALL,
        suffix: str = "one",
        fast: bool = True,
        with_loc: bool = False,
    ):
        """Initialize the Dataset

        Args:
            json_file (Union[str, List[str]]): Path of the file(s) listing the samples
            root_dir (Union[str, List[str]]): _descRoot directories of the SITSription_
            label_dir (str): Directory of the positive samples: binary labels
            transform (transforms.Compose, optional): Data augmentation pipeline. Defaults to None.
            split (str, optional): String indicating the split. Defaults to "train".
            return_paths (bool, optional): Flag to return the sample info. Defaults to False.
            bands (List[str], optional): List of bands to use. Defaults to BANDS_ALL.
            suffix (str, optional): Suffix to access the right npy files. Defaults to "one".
            fast (bool, optional): Flag to access fast loading. Defaults to True.
            with_loc (bool, optional):  Flag to use localizatio. Defaults to False.

        Raises:
            NotImplementedError: Does not support multiple split files
            NotImplementedError: Mixing bands across bands group is not possible
            NotImplementedError: Mixing bands across bands group is not possible
            NotImplementedError: Mixing bands across bands group is not possible
        """

        if isinstance(json_file, str):
            self.data_paths = pd.read_json(json_file, lines=True)[CANADA_REF_COLS]
        if isinstance(json_file, list):
            raise NotImplementedError("Multiple json files not supported yet")

        if split == "train":
            for column, value in TRAIN_FILTER.items():
                if len(value) > 0:
                    self.data_paths = self.data_paths[self.data_paths[column].isin(value)].reset_index(drop=True)
        elif split == "val":
            for column, value in VAL_FILTER.items():
                if len(value) > 0:
                    self.data_paths = self.data_paths[self.data_paths[column].isin(value)].reset_index(drop=True)
        elif split == "test":
            for column, value in TEST_FILTER.items():
                if len(value) > 0:
                    self.data_paths = self.data_paths[self.data_paths[column].isin(value)].reset_index(drop=True)

        if isinstance(root_dir, str):
            self.root_dir = {0: root_dir}
        elif isinstance(root_dir, list):
            self.root_dir = {file_id: root_dir_ for file_id, root_dir_ in enumerate(root_dir)}
        self.label_dir = label_dir
        self.transform = transform
        self.return_paths = return_paths
        self.with_loc = with_loc

        # For __adapt__ method to respect the BANDS_10, BANDS_20, BANDS_60 order in the normalization
        self.bands = []
        if all([band in bands for band in BANDS_10]):
            self.band_10x = True
            self.bands.extend(BANDS_10)
        elif not any([band in bands for band in BANDS_10]):
            self.band_10x = False
        else:
            # self.band_10x = [BANDS_10.index(band) for band in bands if band in BANDS_10]
            raise NotImplementedError("Mixing bands not supported yet")

        if all([band in bands for band in BANDS_20]):
            self.band_20x = True
            self.bands.extend(BANDS_20)
        elif not any([band in bands for band in BANDS_20]):
            self.band_20x = False
        else:
            # self.band_20x = [BANDS_20.index(band) for band in bands if band in BANDS_20]
            raise NotImplementedError("Mixing bands not supported yet")

        if all([band in bands for band in BANDS_60]):
            self.band_60x = True
            self.bands.extend(BANDS_60)
        elif not any([band in bands for band in BANDS_60]):
            self.band_60x = False
        else:
            # self.band_60x = [BANDS_60.index(band) for band in bands if band in BANDS_60]
            raise NotImplementedError("Mixing bands not supported yet")

        self.suffix = suffix
        self.fast = fast

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):

        # If idx is a tensor, convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.fast:
            sample, img_name_info = self.__fastadapt__(idx)
        else:
            sample, img_name_info = self.__adapt__(idx)

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_name_info

        return sample

    def __adapt__(self, idx) -> Tuple[Dict[str, np.ndarray], str]:
        raise NotImplementedError("Adapt method not implemented yet")

    def __fastadapt__(self, idx) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:

        # Extracting key information on S2 SITS
        tile_dir = str(self.data_paths.loc[idx, "tile_id"])
        region_dir = self.data_paths.loc[idx, "region"]
        file_id = self.data_paths.loc[idx, "file_id"]
        fwi = self.data_paths.loc[idx, "fwinx_mean"]

        # Equivalent to Img Name in PASTIS
        img_name_info = {"region": region_dir, "file_id": file_id, "tile_id": tile_dir, "fwinx_mean": fwi}
        img_name_path = os.path.join(self.root_dir[file_id], region_dir, tile_dir)

        images_ts_10 = (
            np.load(os.path.join(img_name_path, f"images_ts_10_{self.suffix}.npy")) if self.band_10x else None
        )
        images_ts_20 = (
            np.load(os.path.join(img_name_path, f"images_ts_20_{self.suffix}.npy")) if self.band_20x else None
        )
        images_ts_60 = (
            np.load(os.path.join(img_name_path, f"images_ts_60_{self.suffix}.npy")) if self.band_60x else None
        )
        doy_ts = np.load(os.path.join(img_name_path, f"doy_ts_{self.suffix}.npy"))

        # Loading the labels
        if file_id == 0:
            labels = np.load(os.path.join(self.label_dir, tile_dir, "label.npy"))
        else:
            labels = np.zeros((LABEL_IMG_RES, LABEL_IMG_RES))

        sample = {"10x": images_ts_10, "20x": images_ts_20, "60x": images_ts_60, "labels": labels, "doy": doy_ts}

        if self.with_loc:
            loc = np.load(os.path.join(img_name_path, f"loc_{self.suffix}.npy"))
            sample["loc"] = loc

        return sample, img_name_info

    def read(self, idx: int) -> Dict[str, np.ndarray]:
        """Read single dataset sample corresponding to idx (index number) without any data transform applied"""
        if isinstance(idx, int):
            _, sample = self.__adapt__(idx)
            return sample

        raise NotImplementedError

    @staticmethod
    def _normalize(img_ts: np.ndarray, with_temp: bool = False, epsilon: float = 1e-10) -> np.ndarray:
        # Reminder Img are TCHW
        if not with_temp:
            kid = img_ts - img_ts.min(axis=(2, 3), keepdims=True)
            mom = img_ts.max(axis=(2, 3), keepdims=True) - img_ts.min(axis=(2, 3), keepdims=True)
        else:
            kid = img_ts - img_ts.min(axis=(0, 2, 3), keepdims=True)
            mom = img_ts.max(axis=(0, 2, 3), keepdims=True) - img_ts.min(axis=(0, 2, 3), keepdims=True)

        img_ts = kid / (mom + epsilon)

        return img_ts


def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b["unk_masks"].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
