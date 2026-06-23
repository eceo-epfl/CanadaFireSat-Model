"""
Module dedicated to trackers for SAEs training. Imported and modified from Overcomplete
"""

import torch


import torch
from torch import nn

class DeadCodeTracker(nn.Module):
    def __init__(self, nb_concepts, dead_feature_window):
        super().__init__()
        self.register_buffer(
            "alive_features",
            torch.zeros(nb_concepts, dtype=torch.bool)
        )
        self.register_buffer(
            "n_updates_since_fired",
            torch.zeros(nb_concepts, dtype=torch.int)
        )
        self.dead_feature_window = dead_feature_window

    def update(self, z):
        fired = (z > 0).any(dim=0)

        self.alive_features |= fired

        self.n_updates_since_fired = torch.where(
            fired,
            torch.zeros_like(self.n_updates_since_fired),
            self.n_updates_since_fired + 1
        )

    def get_dead_ratio(self):
        return 1 - self.alive_features.float().mean().item()
