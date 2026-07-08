"""Evaluation Script for Validation, Test, and Test Hard"""

from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from deepsat.metrics.numpy_metrics import get_classification_metrics
from src.constants import CONFIG_PATH
from src.data import get_data
from src.data.utils import segmentation_ground_truths
from src.eval.utils import get_pr_auc_scores
from src.models.module_img import ImgModule
from src.models.module_tab import TabModule


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval")
@torch.no_grad()
def evaluate(cfg: DictConfig):
    """Evaluation Endpoint"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    seed_everything(cfg["seed"])

    # Load the model
    if cfg["mode"] == "image":
        try:
            model = ImgModule.load_from_checkpoint(cfg["model_path"])
        except (KeyError, RuntimeError):
            with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            model = ImgModule(model_cfg)
            mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
            print("Missing keys:", mis_keys)

        # Test different image sizes
        if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
            model.model.out_H = cfg["MODEL"]["out_H"]
            model.model.out_W = cfg["MODEL"]["out_W"]

        if "ViT" in model.model_type:
            if model.model.features.patch_embed.img_size != (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"]):
                model.model.features.patch_embed.img_size = (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"])
    else:
        try:
            model = TabModule.load_from_checkpoint(cfg["model_path"])
        except KeyError or RuntimeError:
            with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            model = TabModule(model_cfg)
            mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
            print("Missing keys:", mis_keys)
            print("Unexpected keys:", un_keys)

        # Test different image sizes
        if model.model_type in ["TabTSViT", "TabConvLSTM", "UNet", "UTAE"]:

            if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
                model.model.out_H = cfg["MODEL"]["out_H"]
                model.model.out_W = cfg["MODEL"]["out_W"]

        elif model.model_type in ["EnvResNet", "EnvViTFactorizeModel"]:
            if (
                cfg["MODEL"]["out_H"] != model.model.mid_model.out_H
                or cfg["MODEL"]["out_W"] != model.model.mid_model.out_W
            ):
                model.model.mid_model.out_H = cfg["MODEL"]["out_H"]
                model.model.mid_model.out_W = cfg["MODEL"]["out_W"]

        else:
            if (
                cfg["MODEL"]["out_H"] != model.model.sat_model.out_H
                or cfg["MODEL"]["out_W"] != model.model.sat_model.out_W
            ):
                model.model.sat_model.out_H = cfg["MODEL"]["out_H"]
                model.model.sat_model.out_W = cfg["MODEL"]["out_W"]

        if "ViT" in model.model_type and model.model_type != "EnvViTFactorizeModel":

            if model.model.sat_model.features.patch_embed.img_size != (
                cfg["MODEL"]["img_res"],
                cfg["MODEL"]["img_res"],
            ):
                model.model.sat_model.features.patch_embed.img_size = (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"])

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    # Load the dataset need to extend a possibility of no augmentation
    datamodule = get_data(cfg)
    dataset = datamodule.test_dataloader(split=cfg["split"]).dataset

    # Create output directory
    if "test_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["test_max_seq_len"])
    elif "env_val_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["env_val_max_seq_len"])
    else:
        temp_size = "adapt"

    temp_size = (
        temp_size + f"_{cfg['DATASETS']['kwargs']['eval_sampling']}"
        if "eval_sampling" in cfg["DATASETS"]["kwargs"]
        else temp_size
    )
    spa_size = str(cfg["MODEL"]["img_res"]) if "img_res" in cfg["MODEL"] else str(cfg["MODEL"]["mid_input_res"])

    if cfg["DATASETS"]["eval"].get("hard"):
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}_hard"
    else:
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Outputs initialization
    tot_preds = []
    tot_labels = []
    tot_losses = []
    tot_probs = []

    # Collect metadata
    tot_regions = []
    tot_fwi = []

    # Evaluation loop
    for i in tqdm(range(len(dataset)), desc=f"Inferring on split {cfg['split']}", total=len(dataset)):
        data = dataset[i]
        # Extract Batch & Forward Pass
        with torch.no_grad():
            if cfg["mode"] == "image":
                sample = data[0]
                img_name_info = data[1]
                logits = model(sample["inputs"].unsqueeze(0).to(device))
            else:

                if model.model_type in [
                    "TabTSViT",
                    "TabConvLSTM",
                    "TabResNetConvLSTM",
                    "TabViTFactorizeModel",
                    "MultiViTFactorizeModel",
                ]:

                    sample = data[0]
                    tab_sample = data[1]
                    img_name_info = data[2]
                    logits = model(
                        sample["inputs"].unsqueeze(0).to(device),
                        tab_sample["tab_inputs"].unsqueeze(0).to(device),
                        tab_sample["mask"].unsqueeze(0).to(device),
                    )

                elif model.model_type in ["EnvResNet", "EnvViTFactorizeModel", "UNet", "UTAE"]:

                    sample = data[0]
                    img_name_info = data[1]
                    logits = model(
                        xmid=sample["mid_inputs"].unsqueeze(0).to(device),
                        xlow=sample["low_inputs"].unsqueeze(0).to(device),
                        m_mid=sample["mid_inputs_mask"].unsqueeze(0).to(device),
                        m_low=sample["low_inputs_mask"].unsqueeze(0).to(device),
                    )

                else:

                    sample = data[0]
                    env_sample = data[1]
                    img_name_info = data[2]

                    logits = model(
                        x=sample["inputs"].unsqueeze(0).to(device),
                        xmid=env_sample["mid_inputs"].unsqueeze(0).to(device),
                        xlow=env_sample["low_inputs"].unsqueeze(0).to(device),
                        m_mid=env_sample["mid_inputs_mask"].unsqueeze(0).to(device),
                        m_low=env_sample["low_inputs_mask"].unsqueeze(0).to(device),
                    )

        logits = logits.permute(0, 2, 3, 1)
        ground_truth = segmentation_ground_truths(sample)
        labels, unk_masks = ground_truth

        if cfg["MODEL"]["num_classes"] == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > cfg["MODEL"]["threshold"]).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)

        loss = model.loss_fn["mean"](
            logits.reshape(-1, cfg["MODEL"]["num_classes"]), ground_truth[0].to(device).reshape(-1).long()
        )

        if unk_masks is not None:
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            probs = probs.view(-1, cfg["MODEL"]["num_classes"])[unk_masks.view(-1)].cpu().numpy()
            labels = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            preds = predicted.view(-1).cpu().numpy()
            probs = probs.view(-1, cfg["MODEL"]["num_classes"]).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()

        loss = loss.view(-1).cpu().detach().numpy()

        tot_preds.append(preds)
        tot_labels.append(labels)
        tot_losses.append(loss)
        tot_probs.append(probs)

        region = [img_name_info["region"]] * len(preds)
        fwi = [img_name_info["fwinx_mean"]] * len(preds)
        # fwi = [1.]*len(preds)
        tot_regions.append(region)
        tot_fwi.append(fwi)

    # Concatenate all predictions
    predicted_classes = np.concatenate(tot_preds)
    target_classes = np.concatenate(tot_labels)
    losses = np.concatenate(tot_losses)
    print(losses.shape)
    probs_classes = np.concatenate(tot_probs)
    np.save(output_dir / f"{cfg['split']}_probs.npy", probs_classes)
    np.save(output_dir / f"{cfg['split']}_target.npy", target_classes)
    np.save(output_dir / f"{cfg['split']}_preds.npy", predicted_classes)
    np.save(output_dir / f"{cfg['split']}_losses_debug.npy", predicted_classes)

    # Concatenate all metadata
    regions = np.concatenate(tot_regions)
    fwis = np.concatenate(tot_fwi)
    np.save(output_dir / f"{cfg['split']}_regions.npy", regions)
    np.save(output_dir / f"{cfg['split']}_fwi.npy", fwis)

    # Compute metrics
    eval_metrics = get_classification_metrics(
        predicted=predicted_classes,
        labels=target_classes,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        unk_masks=None,
    )

    micro_auc, macro_auc, class_auc = get_pr_auc_scores(
        scores=probs_classes,
        labels=target_classes,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        output_dir=output_dir,
    )

    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics["micro"]
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics["macro"]
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]

    metrics = {
        "loss": losses.mean(),
        "macro_Accuracy": macro_acc,
        "macro_Precision": macro_precision,
        "macro_Recall": macro_recall,
        "macro_F1": macro_F1,
        "macro_IOU": macro_IOU,
        "macro_AUC": macro_auc,
        "micro_Accuracy": micro_acc,
        "micro_Precision": micro_precision,
        "micro_Recall": micro_recall,
        "micro_F1": micro_F1,
        "micro_IOU": micro_IOU,
        "micro_AUC": micro_auc,
        "fire_Accuracy": class_acc[1],
        "fire_Precision": class_precision[1],
        "fire_Recall": class_recall[1],
        "fire_F1": class_F1[1],
        "fire_IOU": class_IOU[1],
        "fire_AUC": class_auc[1],
    }

    with open(output_dir / f"{cfg['split']}_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    if cfg["with_region"]:
        for region in np.unique(regions):
            idx = np.where(regions == region)

            region_preds = predicted_classes[idx]
            region_labels = target_classes[idx]
            region_probs = probs_classes[idx]

            region_metrics = get_classification_metrics(
                predicted=region_preds,
                labels=region_labels,
                n_classes=(
                    cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
                ),
                unk_masks=None,
            )
            region_micro_auc, region_macro_auc, region_class_auc = get_pr_auc_scores(
                scores=region_probs,
                labels=region_labels,
                n_classes=(
                    cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
                ),
            )

            region_micro_acc, region_micro_precision, region_micro_recall, region_micro_F1, region_micro_IOU = (
                region_metrics["micro"]
            )
            region_macro_acc, region_macro_precision, region_macro_recall, region_macro_F1, region_macro_IOU = (
                region_metrics["macro"]
            )
            region_class_acc, region_class_precision, region_class_recall, region_class_F1, region_class_IOU = (
                region_metrics["class"]
            )

            region_metrics = {
                "macro_Accuracy": region_macro_acc,
                "macro_Precision": region_macro_precision,
                "macro_Recall": region_macro_recall,
                "macro_F1": region_macro_F1,
                "macro_IOU": region_macro_IOU,
                "macro_AUC": region_macro_auc,
                "micro_Accuracy": region_micro_acc,
                "micro_Precision": region_micro_precision,
                "micro_Recall": region_micro_recall,
                "micro_F1": region_micro_F1,
                "micro_IOU": region_micro_IOU,
                "micro_AUC": region_micro_auc,
                "fire_Accuracy": region_class_acc[1],
                "fire_Precision": region_class_precision[1],
                "fire_Recall": region_class_recall[1],
                "fire_F1": region_class_F1[1],
                "fire_IOU": region_class_IOU[1],
                "fire_AUC": region_class_auc[1],
            }

            with open(output_dir / f"{cfg['split']}_{region}_metrics.txt", "w") as f:
                for key, value in region_metrics.items():
                    f.write(f"{key}: {value}\n")

    print("Evaluation completed!")
    print(metrics)


if __name__ == "__main__":
    evaluate()
