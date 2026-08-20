from typing import Any, Dict

import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import Dataset

from src.data.utils import segmentation_ground_truths


class NpyActivationDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, mmap_mode="r")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.data[idx].copy())


"""
class OnlineActivationDataset(Dataset):
    def __init__(self, model: LightningModule, input_dataset: Dataset, use_temp: bool = True, device: str = "cuda"):
        self.model = model.eval()
        self.input_dataset = input_dataset
        self.use_temp = use_temp
        self.device = device
        self.size = (self.model.model.num_patches**0.5, self.model.model.num_patches**0.5)

    def __len__(self) -> int:
        return len(self.input_dataset) * self.model.model.num_patches

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            data = self.input_dataset[idx]
            sample = data[0]
            patch_embed = self.model.model.encode_patches(sample["inputs"].unsqueeze(0).to(self.device), use_temp=True,
                                                          doy=sample["doy"].unsqueeze(0).to(self.device), seq_len=sample["seq_lengths"])
            patch_embed = patch_embed.squeeze(0) # [P, D]
            ground_truth = segmentation_ground_truths(sample)
            labels, unk_masks = ground_truth

            if labels.dim() == 2:
                labels = labels.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            elif labels.dim() == 3:
                labels = labels.unsqueeze(0) # [1, 1, H, W]

            labels = torch.nn.functional.interpolate(labels.float(), size=self.size, mode="nearest")
            labels = labels.round().long()

            if unk_masks is not None:

                if unk_masks.dim() == 2:
                    unk_masks = unk_masks.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
                elif unk_masks.dim() == 3:
                    unk_masks = unk_masks.unsqueeze(0) # [1, 1, H, W]

                unk_masks = torch.nn.functional.interpolate(unk_masks.float(), size=self.size, mode="nearest")
                unk_masks = unk_masks.round().long()
                labels = labels.view(-1)[unk_masks.view(-1)]
                patch_embed = patch_embed[unk_masks.view(-1), :]

            else:
                labels = labels.view(-1)

            return {
                "data": patch_embed,
                "label": labels,
            }
        """

class OnlineActivationDataset(Dataset):
    def __init__(self, input_dataset: Dataset):
        self.input_dataset = input_dataset

    def __len__(self) -> int:
        return len(self.input_dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        data = self.input_dataset[idx]
        sample = data[0]
        ground_truth = segmentation_ground_truths(sample)
        labels, unk_masks = ground_truth
        return {
            "inputs": sample["inputs"],           # [T=5, C, H, W], fixed
            "doy": sample["doy"],                 # [T=5]
            "seq_lengths": sample["seq_lengths"],
            "labels": labels,
            "unk_masks": unk_masks,
        }
