"""Library of lightning datamodules for different data type"""

from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule

from src.constants import ENV_SOURCE_COLS, PREPROCESSING_SUFFIX, TAB_SOURCE_COLS
from src.data.augmentations import MixHVFlip
from src.data.Canada.data_transforms import (
    Canada_segmentation_transform,
    EnvBaselineCanada_segmentation_transform,
    EnvCanada_segmentation_transform,
    TabCanada_segmentation_transform,
)
from src.data.Canada.dataloader import get_dataloader
from src.data.Canada.env_dataloader import get_dataloader as get_env_dataloader
from src.data.Canada.mix_dataloader import get_dataloader as get_tab_dataloader


class SatDataModule(LightningDataModule):
    """Datamodule for the SITS"""

    def __init__(
        self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs
    ):
        super().__init__()

        # Train key variables
        self.train_paths_file = train_config["paths"]
        self.train_batch_size = train_config["batch_size"]
        self.train_num_workers = train_config["num_workers"]
        self.train_root_dir = train_config["base_dir"]
        self.train_label_dir = train_config["label_dir"]
        self.train_transform = Canada_segmentation_transform(model_config=model_config, is_training=True, **kwargs)
        self.target_file_id = train_config["target_file_id"] if "target_file_id" in train_config else None
        self.fwi_th = (
            train_config["fwi_ths"][0] if ("fwi_ths" in train_config) and (0 in train_config["fwi_ths"]) else None
        )

        # Validation key variables
        self.val_paths_file = eval_config["paths"]
        self.val_batch_size = eval_config["batch_size"]
        self.val_num_workers = eval_config["num_workers"]
        self.val_root_dir = eval_config["base_dir"]
        self.val_label_dir = eval_config["label_dir"]
        self.val_transform = Canada_segmentation_transform(model_config=model_config, is_training=False, **kwargs)

        self.suffix = PREPROCESSING_SUFFIX
        self.kwargs = kwargs
        self.test_transform = Canada_segmentation_transform(
            model_config=model_config, is_training=False, is_eval=True, **kwargs
        )

    def __len__(self):
        return len(self.train_dataloader().dataset)

    def train_dataloader(self, target_file_id: Optional[int] = None, fwi_th: Optional[float] = None):

        if target_file_id is not None:
            _target_file_id = target_file_id
        else:
            _target_file_id = self.target_file_id

        if fwi_th is not None:
            _fwi_th = fwi_th
        else:
            _fwi_th = self.fwi_th

        return get_dataloader(
            paths_file=self.train_paths_file,
            root_dir=self.train_root_dir,
            label_dir=self.train_label_dir,
            split="train",
            transform=self.train_transform,
            fast=True,
            suffix=self.suffix,
            batch_size=self.train_batch_size,
            fwi_th=_fwi_th,
            target_file_id=_target_file_id,
            num_workers=self.train_num_workers,
            **self.kwargs,
        )

    def val_dataloader(self):
        return get_dataloader(
            paths_file=self.val_paths_file,
            root_dir=self.val_root_dir,
            label_dir=self.val_label_dir,
            split="val",
            transform=self.val_transform,
            fast=True,
            suffix=self.suffix,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            **self.kwargs,
        )

    def test_dataloader(self, split: str = "test"):
        # Files path should be the same for test and val
        return get_dataloader(
            paths_file=self.val_paths_file,
            root_dir=self.val_root_dir,
            label_dir=self.val_label_dir,
            split=split,
            transform=self.test_transform,
            fast=True,
            suffix=self.suffix,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            return_paths=True,
            **self.kwargs,
        )


class TabSatDataModule(LightningDataModule):
    """Datamodule for Mix data"""

    def __init__(
        self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs
    ):
        super().__init__()

        # Train key variables
        self.train_paths_file = train_config["paths"]
        self.train_batch_size = train_config["batch_size"]
        self.train_num_workers = train_config["num_workers"]
        self.train_root_dir = train_config["base_dir"]
        self.train_label_dir = train_config["label_dir"]
        if "is_spatial" in kwargs and kwargs["is_spatial"]:
            self.train_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=True, img_only=False, **kwargs
            )
            self.tab_train_transform = EnvCanada_segmentation_transform(
                model_config=model_config, is_training=True, env_only=False, **kwargs
            )
            self.mix_train_transform = MixHVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=kwargs["with_loc"])
            self.tab_train_cols = ENV_SOURCE_COLS
        else:
            self.train_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=True, img_only=True, **kwargs
            )
            self.tab_train_transform = TabCanada_segmentation_transform(
                model_config=model_config, is_training=True, **kwargs
            )
            self.mix_train_transform = None
            self.tab_train_cols = TAB_SOURCE_COLS
        self.target_file_id = train_config["target_file_id"] if "target_file_id" in train_config else None
        self.fwi_th = (
            train_config["fwi_ths"][0] if ("fwi_ths" in train_config) and (0 in train_config["fwi_ths"]) else None
        )

        # Eval key variables
        self.val_paths_file = eval_config["paths"]
        self.val_batch_size = eval_config["batch_size"]
        self.val_num_workers = eval_config["num_workers"]
        self.val_root_dir = eval_config["base_dir"]
        self.val_label_dir = eval_config["label_dir"]
        if "is_spatial" in kwargs and kwargs["is_spatial"]:
            self.val_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=False, img_only=False, **kwargs
            )
            self.tab_val_transform = EnvCanada_segmentation_transform(
                model_config=model_config, is_training=False, env_only=False, **kwargs
            )

            self.tab_val_cols = ENV_SOURCE_COLS
        else:
            self.val_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=False, img_only=True, **kwargs
            )
            self.tab_val_transform = TabCanada_segmentation_transform(
                model_config=model_config, is_training=False, **kwargs
            )
            self.tab_val_cols = TAB_SOURCE_COLS
        self.suffix = PREPROCESSING_SUFFIX
        self.kwargs = kwargs
        self.test_transform = Canada_segmentation_transform(
            model_config=model_config, is_training=False, is_eval=True, **kwargs
        )

    def __len__(self):
        return len(self.train_dataloader().dataset)

    def train_dataloader(self, target_file_id: Optional[int] = None, fwi_th: Optional[float] = None):

        if target_file_id is not None:
            _target_file_id = target_file_id
        else:
            _target_file_id = self.target_file_id

        if fwi_th is not None:
            _fwi_th = fwi_th
        else:
            _fwi_th = self.fwi_th

        return get_tab_dataloader(
            paths_file=self.train_paths_file,
            root_dir=self.train_root_dir,
            label_dir=self.train_label_dir,
            split="train",
            transform=self.train_transform,
            tab_transform=self.tab_train_transform,
            mix_transform=self.mix_train_transform,
            fast=True,
            suffix=self.suffix,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            target_file_id=_target_file_id,
            fwi_th=_fwi_th,
            tab_source_cols=self.tab_train_cols,
            **self.kwargs,
        )

    def val_dataloader(self):
        return get_tab_dataloader(
            paths_file=self.val_paths_file,
            root_dir=self.val_root_dir,
            label_dir=self.val_label_dir,
            split="val",
            transform=self.val_transform,
            tab_transform=self.tab_val_transform,
            fast=True,
            suffix=self.suffix,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )

    def test_dataloader(self, split: str = "test"):
        return get_tab_dataloader(
            paths_file=self.val_paths_file,
            root_dir=self.val_root_dir,
            label_dir=self.val_label_dir,
            split=split,
            transform=self.test_transform,
            tab_transform=self.tab_val_transform,
            fast=True,
            suffix=self.suffix,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            return_paths=True,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )


class EnvDataModule(LightningDataModule):
    """Datamodule for ENV only data"""

    def __init__(
        self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs
    ):
        super().__init__()

        # Train key variables
        self.train_paths_file = train_config["paths"]
        self.train_batch_size = train_config["batch_size"]
        self.train_num_workers = train_config["num_workers"]
        self.train_label_dir = train_config["label_dir"]
        self.tab_train_transform = (
            EnvCanada_segmentation_transform(model_config=model_config, env_only=True, **kwargs)
            if model_config["architecture"] not in ["UNet", "UTAE"]
            else EnvBaselineCanada_segmentation_transform(model_config=model_config, env_only=True, **kwargs)
        )
        self.tab_train_cols = ENV_SOURCE_COLS
        self.target_file_id = train_config["target_file_id"] if "target_file_id" in train_config else None
        self.fwi_th = (
            train_config["fwi_ths"][0] if ("fwi_ths" in train_config) and (0 in train_config["fwi_ths"]) else None
        )

        # Eval key variables
        self.val_paths_file = eval_config["paths"]
        self.val_batch_size = eval_config["batch_size"]
        self.val_num_workers = eval_config["num_workers"]
        self.val_label_dir = eval_config["label_dir"]
        self.tab_val_transform = (
            EnvCanada_segmentation_transform(model_config=model_config, env_only=True, is_training=False, **kwargs)
            if model_config["architecture"] not in ["UNet", "UTAE"]
            else EnvBaselineCanada_segmentation_transform(
                model_config=model_config, env_only=True, is_training=False, **kwargs
            )
        )
        self.tab_val_cols = ENV_SOURCE_COLS
        self.suffix = PREPROCESSING_SUFFIX
        self.kwargs = kwargs

    def __len__(self):
        return len(self.train_dataloader().dataset)

    def train_dataloader(self, target_file_id: Optional[int] = None, fwi_th: Optional[float] = None):

        if target_file_id is not None:
            _target_file_id = target_file_id
        else:
            _target_file_id = self.target_file_id

        if fwi_th is not None:
            _fwi_th = fwi_th
        else:
            _fwi_th = self.fwi_th

        return get_env_dataloader(
            paths_file=self.train_paths_file,
            label_dir=self.train_label_dir,
            split="train",
            tab_transform=self.tab_train_transform,
            suffix=self.suffix,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            target_file_id=_target_file_id,
            fwi_th=_fwi_th,
            tab_source_cols=self.tab_train_cols,
            **self.kwargs,
        )

    def val_dataloader(self):
        return get_env_dataloader(
            paths_file=self.val_paths_file,
            label_dir=self.val_label_dir,
            split="val",
            tab_transform=self.tab_val_transform,
            suffix=self.suffix,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )

    def test_dataloader(self, split: str = "test"):
        return get_env_dataloader(
            paths_file=self.val_paths_file,
            root_dir=self.val_root_dir,
            label_dir=self.val_label_dir,
            split=split,
            tab_transform=self.tab_val_transform,
            suffix=self.suffix,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            return_paths=True,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )
