from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments.concept_bottleneck.sae.dataset import NpyActivationDataset, OnlineActivationDataset


class ActDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_features: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        test_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features

    def train_dataloader(self, num_workers: int = 8) -> DataLoader:
        return DataLoader(
            TensorDataset(self.train_features), batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

    def val_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.val_features is not None:
            return DataLoader(
                TensorDataset(self.val_features), batch_size=self.batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            return []

    def test_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.test_features is not None:
            return DataLoader(
                TensorDataset(self.test_features), batch_size=self.batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            return []


class NpyActDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_npy_path: str,
        val_npy_path: Optional[str] = None,
        test_npy_path: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_npy_path = train_npy_path
        self.val_npy_path = val_npy_path
        self.test_npy_path = test_npy_path

    def train_dataloader(self, num_workers: int = 8) -> DataLoader:
        dataset = NpyActivationDataset(self.train_npy_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.val_npy_path is not None:
            dataset = NpyActivationDataset(self.val_npy_path)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            return []

    def test_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.test_npy_path is not None:
            dataset = NpyActivationDataset(self.test_npy_path)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            return []

"""
class OnlineActDataModule(LightningDataModule):
    def __init__(
        self,
        model: LightningModule,
        batch_size: int,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device

    def train_dataloader(self, num_workers: int = 8) -> DataLoader:
        return DataLoader(
            OnlineActivationDataset(self.model, self.train_dataset, device=self.device), batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

    def val_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.val_dataset is not None:
            return DataLoader(
                OnlineActivationDataset(self.model, self.val_dataset, device=self.device), batch_size=self.batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            return []

    def test_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.test_dataset is not None:
            return DataLoader(
                OnlineActivationDataset(self.model, self.test_dataset, device=self.device), batch_size=self.batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            return []"""


class OnlineActDataModule(LightningDataModule):
    def __init__(
        self,
        model: LightningModule,
        batch_size: int,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        device: str = "cuda",
        test_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model = model.eval()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self._device_str = device
        self.size = (
            int(self.model.model.num_patches**0.5),
            int(self.model.model.num_patches**0.5),
        )

    def setup(self, stage: Optional[str] = None):
        self.model = self.model.to(self._device_str)

    def train_dataloader(self, num_workers: int = 8) -> DataLoader:
        return DataLoader(
            OnlineActivationDataset(self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def val_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.val_dataset is not None:
            return DataLoader(
                OnlineActivationDataset(self.val_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        return []

    def test_dataloader(self, num_workers: int = 8) -> Union[DataLoader, List]:
        if self.test_dataset is not None:
            return DataLoader(
                OnlineActivationDataset(self.test_dataset),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        return []

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, torch.Tensor]:
        inputs = batch["inputs"].to(self._device_str)
        doy = batch["doy"].to(self._device_str)
        seq_lengths = batch["seq_lengths"].to(self._device_str)

        # print(f"Batch inputs shape: {inputs.shape}, DOY shape: {doy.shape}, Seq lengths shape: {seq_lengths.shape}")

        with torch.no_grad():
            patch_embed = self.model.model.encode_patches(
                inputs, use_temp=True, doy=doy, seq_len=seq_lengths,
            )  # [B, P, D]
            patch_embed = patch_embed.view(-1, patch_embed.shape[-1])  # [B * P, D]

        # --- Batched label prep ---
        labels = batch["labels"].to(self._device_str)  # [B, ...]

        #print(f"Batch labels shape: {labels.shape}")
        # print(f"Batch patch_embed shape: {patch_embed.shape}")

        if labels.dim() == 3:      # [B, H, W]
            labels = labels.unsqueeze(1)  # [B, 1, H, W]
        elif labels.dim() == 4 and labels.shape[1] != 1:
            labels = labels.unsqueeze(1)  # be explicit; adjust if your raw dim layout differs

        labels = torch.nn.functional.interpolate(
            labels.float(), size=self.size, mode="nearest"
        ).round().long()  # [B, 1, H', W']
        labels = labels.view(-1)  # [B * P]

        # print(f"Resized and flattened labels shape: {labels.shape}")

        ## --- Ignoring unknown masks for now; CanadaFireSat does not contain them ---

        return {
            "data": patch_embed,
            "label": labels,
        }