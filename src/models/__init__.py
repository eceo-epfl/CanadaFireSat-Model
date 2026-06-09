from src.models.module_img import ImgModule
from src.models.module_tab import TabModule


def get_model(config, device):
    if config["MODEL"]["architecture"] in ["TSViT", "ConvLSTM", "ResNet", "ViT", "ViTFacto", "MSClipFacto"]:
        return ImgModule(config).to(device)

    elif config["MODEL"]["architecture"] in [
        "TabTSViT",
        "TabConvLSTM",
        "MixResNet",
        "EnvResNet",
        "EnvViTFactorizeModel",
        "TabResNetConvLSTM",
        "TabViTFactorizeModel",
        "MultiViTFactorizeModel",
    ]:
        return TabModule(config).to(device)

    else:
        raise NameError(
            f"Model architecture {config['MODEL']['architecture']} not found, choose from: 'TSViT' or 'TabTSViT'"
        )
