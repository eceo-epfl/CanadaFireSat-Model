from omegaconf import DictConfig, OmegaConf

from src.data.Canada.datamodule import EnvDataModule, SatDataModule, TabSatDataModule
from src.data.hf_Canada.hf_datamodule import EnvDataModule as HfEnvDataModule
from src.data.hf_Canada.hf_datamodule import SatDataModule as HfSatDataModule
from src.data.hf_Canada.hf_datamodule import TabSatDataModule as HfTabSatDataModule


def get_data(config):

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    kwargs = config["DATASETS"].get("kwargs") or {}

    architecture = config["MODEL"]["architecture"]
    train_config = config["DATASETS"]["train"]
    eval_config = config["DATASETS"]["eval"]

    if architecture in [
        "TabTSViT",
        "TabConvLSTM",
        "MixResNet",
        "TabResNetConvLSTM",
        "TabViTFactorizeModel",
        "MultiViTFactorizeModel",
    ]:
        datamodule_cls = HfTabSatDataModule if config["DATASETS"].get("mode") == "huggingface" else TabSatDataModule
    elif architecture in ["EnvResNet", "EnvViTFactorizeModel"]:
        datamodule_cls = HfEnvDataModule if config["DATASETS"].get("mode") == "huggingface" else EnvDataModule
    else:
        datamodule_cls = HfSatDataModule if config["DATASETS"].get("mode") == "huggingface" else SatDataModule

    return datamodule_cls(model_config=config["MODEL"], train_config=train_config, eval_config=eval_config, **kwargs)
