"""Build Dataloader for multi-modal data from *.npy files (and csv files)"""

from __future__ import division, print_function

import json
import os
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
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
    TAB_SOURCE_COLS,
    TEST_FILTER,
    TRAIN_FILTER,
    VAL_FILTER,
)


def get_dataloader(
    paths_file: Union[str, List[str]],
    root_dir: Union[str, List[str]],
    tab_dir: Union[str, List[str]],
    label_dir: str,
    transform: transforms.Compose = None,
    tab_transform: transforms.Compose = None,
    mix_transform: transforms.Compose = None,
    split: str = "train",
    bands: List[str] = BANDS_ALL,
    suffix: str = "one",
    fast: bool = True,
    tab_source_cols: Dict[str, List[str]] = TAB_SOURCE_COLS,
    nan_value: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    return_paths: bool = False,
    my_collate: Callable = None,
    target_file_id: Optional[int] = None,
    fwi_th: float = None,
    with_loc: bool = False,
    is_spatial: bool = False,
    one_tile: bool = False,
    **kwargs,
):
    """Get Dataloader for Mix data.

    Args:
        paths_file (Union[str, List[str]]): Path of the file(s) listing the samples
        root_dir (Union[str, List[str]]): Root directories of the SITS
        tab_dir (Union[str, List[str]]): Directories of the environmental inputs
        label_dir (str): Directory of the positive samples: binary labels
        transform (transforms.Compose, optional): Data augmentation pipeline SITS. Defaults to None.
        tab_transform (transforms.Compose, optional): Data augmentation pipeline ENV. Defaults to None.
        mix_transform (transforms.Compose, optional): Data augmentation pipeline MIX. Defaults to None.
        split (str, optional): String indicating the split. Defaults to "train".
        bands (List[str], optional): List of bands to use. Defaults to BANDS_ALL.
        suffix (str, optional): Suffix to access the right npy files. Defaults to "one".
        fast (bool, optional): Flag to access fast loading. Defaults to True.
        tab_source_cols (Dict[str, List[str]], optional): Dictionary mapping sources to target variables. Defaults to TAB_SOURCE_COLS.
        nan_value (float, optional): Value to fill missing entries. Defaults to 0.0.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for loading samples. Defaults to 4.
        shuffle (bool, optional): Flag to shuffle samples. Defaults to True.
        return_paths (bool, optional): Flag to return the sample info. Defaults to False.
        my_collate (Callable, optional): Collate function for the batch. Defaults to None.
        target_file_id (Optional[int], optional): File Id to sample only one specific sample type. Defaults to None.
        fwi_th (float, optional): FWI threshild for sampling specific fire risk. Defaults to None.
        with_loc (bool, optional): Flag to use localization. Defaults to False.
        is_spatial (bool, optional): Flag to use spatial inputs. Defaults to False.
        one_tile (bool, optional): Flag to use only one tile for ablation. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: MIX Dataloader
    """

    dataset = SatImTabDataset(
        json_file=paths_file,
        root_dir=root_dir,
        tab_dir=tab_dir,
        label_dir=label_dir,
        transform=transform,
        tab_transform=tab_transform,
        mix_transform=mix_transform,
        split=split,
        return_paths=return_paths,
        bands=bands,
        suffix=suffix,
        fast=fast,
        tab_source_cols=tab_source_cols,
        nan_value=nan_value,
        with_loc=with_loc,
        is_spatial=is_spatial,
        one_tile=one_tile,
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


def process_nan(tab_data: np.ndarray, nan_value: Union[str, float] = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Process NaN values in environmental data"""
    mask = np.isnan(tab_data)
    if isinstance(nan_value, float):
        tab_data = np.nan_to_num(tab_data, nan=nan_value)
    elif nan_value == "median":
        median_mod = np.nanmedian(tab_data, axis=0)
        tab_data = np.nan_to_num(tab_data, nan=median_mod)
        tab_data = np.nan_to_num(tab_data, nan=0.0)
    elif nan_value == "nan":
        pass
    else:
        raise ValueError("nan_value should be either 'median', 'nan' or a float")
    return tab_data, mask


class SatImTabDataset(Dataset):
    """Mix dataset."""

    def __init__(
        self,
        json_file: Union[str, List[str]],
        root_dir: Union[str, List[str]],
        tab_dir: Union[str, List[str]],
        label_dir: str,
        transform: transforms.Compose = None,
        tab_transform: transforms.Compose = None,
        mix_transform: transforms.Compose = None,
        split: str = "train",
        return_paths: bool = False,
        bands: List[str] = BANDS_ALL,
        suffix: str = "one",
        fast: bool = True,
        with_loc: bool = False,
        tab_source_cols: Dict[str, List[str]] = TAB_SOURCE_COLS,
        is_spatial: bool = False,
        nan_value: Union[str, float] = 0.0,
        one_tile: bool = False,
    ):
        """Initialize the Dataset

        Args:
            json_file (Union[str, List[str]]): Path of the file(s) listing the samples
            root_dir (Union[str, List[str]]): Root directories of the SITS (DEPRECATED)
            tab_dir (Union[str, List[str]]): Directories of the environmental inputs
            label_dir (str): Directory of the positive samples: binary labels
            transform (transforms.Compose, optional): Data augmentation pipeline SITS. Defaults to None.
            tab_transform (transforms.Compose, optional): Data augmentation pipeline ENV. Defaults to None.
            mix_transform (transforms.Compose, optional): Data augmentation pipeline MIX. Defaults to None.
            split (str, optional): String indicating the split. Defaults to "train".
            return_paths (bool, optional): Flag to return the sample info. Defaults to False.
            bands (List[str], optional): List of bands to use. Defaults to BANDS_ALL.
            suffix (str, optional): Suffix to access the right npy files. Defaults to "one".
            fast (bool, optional): Flag to access fast loading. Defaults to True.
            with_loc (bool, optional): Flag to use localization. Defaults to False.
            tab_source_cols (Dict[str, List[str]], optional): Dictionary mapping sources to target variables. Defaults to TAB_SOURCE_COLS.
            is_spatial (bool, optional): Flag to use spatial inputs. Defaults to False.
            nan_value (float, optional): Value to fill missing entries. Defaults to 0.0.
            one_tile (bool, optional): Flag to use only one tile for ablation. Defaults to False.

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

        if isinstance(tab_dir, str):
            self.tab_dir = {0: tab_dir}
        elif isinstance(tab_dir, list):
            self.tab_dir = {file_id: tab_dir_ for file_id, tab_dir_ in enumerate(tab_dir)}

        self.label_dir = label_dir
        self.transform = transform
        self.tab_transform = tab_transform
        self.mix_transform = mix_transform
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
        self.tab_source_cols = tab_source_cols
        self.nan_value = nan_value
        self.is_spatial = is_spatial
        self.one_tile = one_tile

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):

        # If idx is a tensor, convert it to a list
        if torch.is_tensor(idx):
            idx = idx.item()

        if self.fast:
            sample, tab_sample, img_name_info = self.__fastadapt__(idx)
        else:
            sample, tab_sample, img_name_info = self.__adapt__(idx)

        if self.transform:
            sample = self.transform(sample)

        if self.tab_transform:
            tab_sample = self.tab_transform(tab_sample)

        if self.mix_transform:
            sample, tab_sample = self.mix_transform(sample, tab_sample)

        if self.return_paths:
            return sample, tab_sample, img_name_info

        return sample, tab_sample

    def __adapt__(self, idx) -> Tuple[Dict[str, np.ndarray], str]:
        raise NotImplementedError("This method is not supported yet")

    def __fastadapt__(self, idx) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:

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

        if self.one_tile:
            images_ts_10 = images_ts_10[-1:] if self.band_10x else None  # T is the first dimension
            images_ts_20 = images_ts_20[-1:] if self.band_20x else None
            images_ts_60 = images_ts_60[-1:] if self.band_60x else None
            doy_ts = doy_ts[-1:]

        # Loading the labels
        if file_id == 0:
            labels = np.load(os.path.join(self.label_dir, tile_dir, "label.npy"))
        else:
            labels = np.zeros((LABEL_IMG_RES, LABEL_IMG_RES))

        sample = {"10x": images_ts_10, "20x": images_ts_20, "60x": images_ts_60, "labels": labels, "doy": doy_ts}

        if not self.is_spatial:
            # Loading the tabular data
            tab_ts = []
            tab_doy_ts = None
            ts_path = os.path.join(self.tab_dir[file_id], tile_dir)
            for source, cols in self.tab_source_cols.items():
                tab_data = pd.read_csv(os.path.join(ts_path, f"{source}.csv"), usecols=cols + ["date"])
                tab_data["date"] = tab_data["date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d"))
                tab_data.sort_values("date", inplace=True)  # Sort the data based on date for LSTM based methods
                doy = tab_data["date"].apply(lambda x: x.timetuple().tm_yday).values

                # DOY should be the same across sources
                if tab_doy_ts is None:
                    tab_doy_ts = doy
                else:
                    assert np.all(tab_doy_ts == doy), "DOY mismatch in tabular data"

                tab_data = tab_data[cols].sort_index(axis=1).values  # This is to align the columns for normalization
                tab_ts.append(tab_data)

            tab_ts = np.concatenate(tab_ts, axis=1)  # T, C
            tab_ts, mask = process_nan(tab_ts, self.nan_value)
            env_sample = {"tab": tab_ts, "tab_doy": tab_doy_ts, "mask": mask}

        else:
            env_sample = {}
            ts_path = os.path.join(self.tab_dir[file_id], tile_dir)
            for source, cols in self.tab_source_cols.items():
                cols = sorted(cols)  # Ensure the order is the same for the normalization
                with open(os.path.join(self.tab_dir[file_id], f"{source}_cols.json"), "r") as f:
                    cols_id = json.load(f)

                idx = [cols_id.index(col) for col in cols]
                tab_data = np.load(os.path.join(ts_path, f"{source}.npy"))[idx]  # Input: C, H, W, T
                tab_data = np.moveaxis(tab_data, -1, 0)  # T, C, H, W
                tab_data, tab_mask = process_nan(tab_data, self.nan_value)
                env_sample[source] = tab_data
                env_sample[f"{source}_mask"] = tab_mask

                if self.with_loc:
                    tab_loc = np.load(os.path.join(ts_path, f"{source}_locs.npy"))  # Input: H, W, 2
                    env_sample[f"{source}_loc"] = tab_loc

            tab_doy_ts = np.load(os.path.join(ts_path, "dates.npy"), allow_pickle=True)  # Input: (T,) with T == 8
            assert np.all(tab_doy_ts[:-1] <= tab_doy_ts[1:]), "Environmental data is not in temporal order"
            tab_doy_ts = np.array([dt.timetuple().tm_yday for dt in tab_doy_ts])
            env_sample["tab_doy"] = tab_doy_ts

        if self.with_loc:
            loc = np.load(os.path.join(img_name_path, f"loc_{self.suffix}.npy"))
            sample["loc"] = loc

        return sample, env_sample, img_name_info

    def read(self, idx, abs=False):
        """Read single dataset sample corresponding to idx (index number) without any data transform applied"""
        if isinstance(idx, int):
            _, sample = self.__adapt__(idx)
            return sample

        raise NotImplementedError
