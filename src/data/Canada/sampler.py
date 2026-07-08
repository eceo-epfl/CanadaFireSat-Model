"""Library of Samplers for specific training curriculum strategy"""

import random

from torch.utils.data import Dataset, Sampler


class FileIDSampler(Sampler):
    """Custom Sampler that only samples items with a specific file_id"""

    def __init__(self, dataset: Dataset, target_file_id: int, shuffle: bool = True):
        self.indices = [idx for idx in range(len(dataset)) if dataset.data_paths.loc[idx, "file_id"] == target_file_id]
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class FWISampler(Sampler):
    """Custom Sampler that only samples items with a FWI below a threshold."""

    def __init__(self, dataset: Dataset, fwi_th: float, target_file_id: int = 0, shuffle: bool = True):
        self.indices = [
            idx
            for idx in range(len(dataset))
            if (
                dataset.data_paths.loc[idx, "fwinx_mean"] < fwi_th
                or dataset.data_paths.loc[idx, "file_id"] == target_file_id
            )
        ]
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
