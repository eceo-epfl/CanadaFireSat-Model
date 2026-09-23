import os
import pandas as pd
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm

from src.constants import LAND_COVER_DICT_IDS

def extend_labels_with_water_mask(label_path: os.PathLike, lc_path: Dict[int, os.PathLike],
                                  split_path: os.PathLike, output_path: os.PathLike):

    name_to_id = {name: i for i, name in LAND_COVER_DICT_IDS.items()}

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_json(split_path, lines=True)

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Prcoessing Train/Val/Test splits"):
        tile_id = str(row["tile_id"])
        file_id = row["file_id"]

        # Loading the label and land cover arrays
        lc_ar =  np.load(Path(lc_path[int(file_id)]) / tile_id / "lc.npy")

        if file_id == 0:
            label_file = Path(label_path) / tile_id / "label.npy"
            label_ar = np.load(label_file)
        else:
            label_ar = np.zeros_like(lc_ar, dtype=np.uint8)

        # Testing proper shape
        assert label_ar.shape == lc_ar.shape, f"Shape mismatch between label and land cover arrays for tile {tile_id}."

        # Creating a water mask
        water_mask = (lc_ar == name_to_id["Water"]).astype(label_ar.dtype)

        # Extending the label array with the water mask
        extended_label_ar = ((label_ar + water_mask) > 0).astype(label_ar.dtype)

        # Saving the extended label array
        output_file = output_path / tile_id / "label.npy"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, extended_label_ar)

if __name__ == "__main__":
    extend_labels_with_water_mask(
        label_path="/media/data/boreal_data/labels_update",
        lc_path={0: "/media/data/boreal_data/LC", 1: "/media/data/boreal_data/NEG_LC"},
        split_path="/media/data/boreal_data/full/COPERNICUS/UPDATE/split_one.json",
        output_path="/media/data/boreal_data/toy_labels"
    )
