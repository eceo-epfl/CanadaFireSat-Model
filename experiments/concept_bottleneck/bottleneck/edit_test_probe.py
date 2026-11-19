from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from experiments.concept_bottleneck.eval.landcover_utils import encode_texts
from experiments.concept_bottleneck.eval.utils import load_img_model, feature_terms_from_vocab
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.metrics import _cosine_similarity_matrix
from src.constants import CONFIG_PATH
from src.data import get_data
from experiments.concept_bottleneck.bottleneck.landcover_selection import selection_score, water_scores
from experiments.concept_bottleneck.bottleneck.probe import SAEProbe
from experiments.concept_bottleneck.sae.utils import get_pylogger


log = get_pylogger(__name__)


@torch.no_grad()
def query_similarity(
    probe: SAEProbe, query_emb: torch.Tensor, source: str, vocab_indices: Optional[np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per alive feature, best cosine to any query term and the position of that term: both [n_alive].

    source="atom" compares the SAE dictionary atom, source="term" its nearest vocabulary term.
    """
    if source == "term":
        if vocab_indices is None:
            raise ValueError('edit.source="term" needs an SAE vocabulary and vocab_csv_path: use "atom"')
        features = probe.label_vocab_emb.float().cpu()[torch.as_tensor(vocab_indices, dtype=torch.long)]
    elif source == "atom":
        features = probe.sae.get_dictionary().detach().float().cpu()[probe.alive_idx.cpu()]
    else:
        raise ValueError(f"Unknown edit.source {source}")
    return _cosine_similarity_matrix(features, query_emb.float().cpu()).max(dim=1)


def select_top_features(similarity: torch.Tensor, candidate_mask: torch.Tensor, top_n: int) -> torch.Tensor:
    """Boolean mask of the top_n candidates by similarity. Ties keep their alive-feature order."""
    candidates = torch.where(candidate_mask)[0]
    order = torch.sort(similarity[candidates], descending=True, stable=True).indices
    edit_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    edit_mask[candidates[order[:top_n]]] = True
    return edit_mask


def select_random_features(candidate_mask: torch.Tensor, top_n: int, seed: int) -> torch.Tensor:
    """Boolean mask of top_n candidates drawn uniformly at random without replacement (baseline)."""
    candidates = torch.where(candidate_mask)[0]
    if top_n > candidates.numel():
        log.warning(f"top_n={top_n} exceeds the {candidates.numel()} candidates: selecting all of them")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(candidates.numel(), generator=generator)
    edit_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    edit_mask[candidates[order[:top_n]]] = True
    return edit_mask


def edited_weight_value(fire_weight: torch.Tensor, mode: str, percentile: float) -> float:
    """Weight given to the edited features: 0, or minus a percentile of |w| over the negative weights."""
    if mode == "zero":
        return 0.0
    if mode == "negative_percentile":
        negative_abs = -fire_weight[fire_weight < 0]
        if negative_abs.numel() == 0:
            raise ValueError("The probe has no negative fire weight to take a percentile from")
        return -torch.quantile(negative_abs, percentile / 100.0).item()
    raise ValueError(f"Unknown edit.mode {mode}")



def evaluate_probe(probe: SAEProbe, datamodule: OnlineActDataModule, device: torch.device, flag_test: bool) -> Dict[str, Any]:
    """Run validation and test as in train_probe.py and return the merged metrics."""
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    val_results = trainer.validate(model=probe, datamodule=datamodule)
    if flag_test:
        test_results = trainer.test(model=probe, datamodule=datamodule)
        return {**val_results[0], **test_results[0]}
    return {**val_results[0]}



@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="eval_edit_toy")
def edit_water_test_performance(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = load_img_model(cfg, device)

    ### For Test we want batch size of 1 and adaptive sequence length (no padding) ###
    assert cfg.MODEL.test_max_seq_len is None, "For evaluation, test_max_seq_len must be None to allow adaptive sequence lengths."
    datamodule = get_data(cfg)
    train_dataset = datamodule.test_dataloader(split="train").dataset
    val_dataset = datamodule.test_dataloader(split="val").dataset
    test_dataset = datamodule.test_dataloader(split="test").dataset
    probe_datamodule = OnlineActDataModule(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        flag_downsample=False,
    )
    probe_datamodule.load_input_norm(cfg.probe.sae_ckpt_path)

    log.info(f"Loaded probe checkpoint from {cfg.probe_checkpoint_path}")
    probe: SAEProbe = hydra.utils.instantiate(cfg.probe)
    probe_checkpoint = torch.load(cfg.probe_checkpoint_path, map_location=device)
    mis_keys, un_keys = probe.load_state_dict(probe_checkpoint["state_dict"], strict=True)
    if mis_keys:
        log.warning(f"Missing keys in probe checkpoint: {mis_keys}")
    if un_keys:
        log.warning(f"Unexpected keys in probe checkpoint: {un_keys}")
    probe.to(device).eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_csv_path = cfg.get("vocab_csv_path") if probe.label_vocab_emb is not None else None
    feature_names, vocab_indices, _ = feature_terms_from_vocab(probe, Path(vocab_csv_path) if vocab_csv_path else None)

    # Cosine between each alive feature (atom or nearest vocab term) and the closest query term
    edit_cfg = cfg.edit
    query_terms = list(edit_cfg.terms)
    prompt_template = edit_cfg.prompt_template
    query_texts = [prompt_template.format(term) for term in query_terms] if prompt_template else query_terms
    query_emb = encode_texts(model, query_texts, device)
    similarity, closest_query = query_similarity(probe, query_emb, edit_cfg.source, vocab_indices)

    # Top-n features among those used by the probe with a positive weight on the fire logit, selected by:
    #   text:      cosine to the query terms
    #   landcover: water score against WorldCover (edit.landcover.score), on the lc_year tiles of the train split
    #   hybrid:    top_n // 2 by text, then the rest by landcover among the features not already selected
    #   random:    uniform draw, as a baseline
    fire_row = 0 if probe.num_classes == 1 else 1
    fire_weight = probe.probe.weight.detach()[fire_row].cpu()
    positive_mask = fire_weight > edit_cfg.min_positive_weight
    selection = edit_cfg.get("selection", "text")
    top_n = int(edit_cfg.top_n)

    lc_stats, lc_score = None, None
    if selection in ("landcover", "hybrid"):
        lc_cfg = edit_cfg.landcover
        lc_stats = water_scores(
            probe, probe_datamodule, train_dataset, lc_cfg, model, cfg.MODEL.input_img_res, cfg.MODEL.img_res,
            device, output_dir,
        )
        lc_score = selection_score(lc_stats, lc_cfg.score, int(lc_cfg.min_active_tokens))
        lc_candidates = positive_mask & torch.isfinite(lc_score)

    selected_by = np.full(probe.n_alive, "", dtype=object)
    if selection == "text":
        edit_mask = select_top_features(similarity, positive_mask, top_n)
        selected_by[edit_mask.numpy()] = "text"
    elif selection == "landcover":
        edit_mask = select_top_features(lc_score, lc_candidates, top_n)
        selected_by[edit_mask.numpy()] = "landcover"
    elif selection == "hybrid":
        text_mask = select_top_features(similarity, positive_mask, top_n // 2)
        lc_mask = select_top_features(lc_score, lc_candidates & ~text_mask, top_n - top_n // 2)
        edit_mask = text_mask | lc_mask
        selected_by[text_mask.numpy()] = "text"
        selected_by[lc_mask.numpy()] = "landcover"
    elif selection == "random":
        random_seed = edit_cfg.get("random_seed")
        if random_seed is None:
            random_seed = cfg.get("seed") or 0
        edit_mask = select_random_features(positive_mask, top_n, random_seed)
        selected_by[edit_mask.numpy()] = "random"
    else:
        raise ValueError(f"Unknown edit.selection {selection}")
    if int(edit_mask.sum()) < top_n:
        log.warning(f"Only {int(edit_mask.sum())}/{top_n} features could be selected")
    new_weight = edited_weight_value(fire_weight, edit_cfg.mode, edit_cfg.percentile)

    alive_feature_indices = probe.alive_idx.detach().cpu()
    edit_positions = torch.where(edit_mask)[0]
    edited_features = pd.DataFrame(
        {
            "alive_position": edit_positions.numpy(),
            "sae_feature_index": alive_feature_indices[edit_mask].numpy(),
            "selected_by": selected_by[edit_mask.numpy()],
            "term": [feature_names[i] for i in edit_positions.tolist()],
            "closest_query": [query_terms[i] for i in closest_query[edit_mask].tolist()],
            "cosine": similarity[edit_mask].numpy(),
            "original_weight": fire_weight[edit_mask].numpy(),
            "new_weight": new_weight,
        }
    )
    # Score that selected each feature: cosine for text, the land-cover score for landcover, NaN for random
    group = edited_features["selected_by"].to_numpy()
    selection_values = np.full(len(edited_features), np.nan)
    selection_values[group == "text"] = edited_features["cosine"].to_numpy()[group == "text"]
    if lc_score is not None:
        selection_values[group == "landcover"] = lc_score[edit_positions].numpy()[group == "landcover"]
    edited_features.insert(3, "selection_score", selection_values)
    if lc_stats is not None:  # Water scores of every edited feature, whatever selected it
        edited_features = pd.concat(
            [edited_features, lc_stats.iloc[edit_positions.numpy()].reset_index(drop=True)], axis=1
        )
    # Text group first, then landcover; each sorted by its own selection score
    edited_features = edited_features.sort_values(["selected_by", "selection_score"], ascending=[False, False])

    mode_tag = "zero" if edit_cfg.mode == "zero" else f"neg_p{edit_cfg.percentile:g}"
    min_weight_tag = f"_minw{edit_cfg.min_positive_weight:g}" if edit_cfg.min_positive_weight > 0 else ""
    prompt_tag = "_prompt" if prompt_template else ""
    text_tag = f"{edit_cfg.source}{prompt_tag}"
    if selection in ("landcover", "hybrid"):
        lc_cfg = edit_cfg.landcover
        lc_tag = f"lc_{lc_cfg.score}" + (f"_p{lc_cfg.percentile:g}" if lc_cfg.score == "precision" else "")
    selection_tag = {
        "text": lambda: text_tag,
        "landcover": lambda: lc_tag,
        "hybrid": lambda: f"hybrid_{text_tag}_{lc_tag}",
        "random": lambda: f"random_seed{random_seed}",
    }[selection]()
    run_tag = f"{selection_tag}_top{top_n}{min_weight_tag}_{mode_tag}"
    edited_features.to_csv(output_dir / f"water_edited_features_{run_tag}.csv", index=False)
    if lc_stats is not None:
        lc_stats.assign(
            sae_feature_index=alive_feature_indices.numpy(), fire_weight=fire_weight.numpy(), cosine=similarity.numpy()
        ).to_csv(output_dir / f"water_lc_feature_stats_{lc_tag}.csv", index_label="alive_position")
    log.info(
        f"{int(positive_mask.sum())} features used with positive weight, {int(edit_mask.sum())} selected "
        f"({selection}), new weight {new_weight:.6f}:\n{edited_features.to_string(index=False)}"
    )

    # Overwrite the probe weights of the selected features on the fire logit
    with torch.no_grad():
        probe.probe.weight[fire_row, edit_mask.to(probe.probe.weight.device)] = new_weight

    log.info("Evaluating the water-edited probe...")
    results = evaluate_probe(probe, probe_datamodule, device, cfg.get("flag_test", True))

    results_df = pd.Series(results, name="water_edited").to_frame()
    results_df.to_csv(output_dir / f"water_edited_metrics_{run_tag}.csv")
    log.info(f"Water-edited probe metrics:\n{results_df.to_string()}")


if __name__ == "__main__":
    edit_water_test_performance()
