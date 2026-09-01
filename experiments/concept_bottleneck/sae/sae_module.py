import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from overcomplete.metrics import avg_l2_loss, hoyer, l0_eps
from overcomplete.sae import (
    SAE,
    BatchTopKSAE,
    JumpSAE,
    RelaxedArchetypalDictionary,
    TopKSAE,
)
from overcomplete.sae.train import _compute_reconstruction_error, extract_input
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
from torchmetrics import R2Score

from experiments.concept_bottleneck.sae.metrics import (
    compute_coherence,
    compute_effective_rank,
    compute_stable_rank,
)
from experiments.concept_bottleneck.sae.sae_utils import (
    criterion_factory,
    mse_criterion,
    optimizer_factory,
    scheduler_factory,
)
from experiments.concept_bottleneck.sae.trackers import DeadCodeTracker
from experiments.concept_bottleneck.sae.utils import points_ext, points_vocab

NAME_CLASS = {0: "No Fire", 1: "Fire"}


class plSAE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        sae_type: str = "topk",
        loss_type: str = "mse",
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = None,
        num_samples: int = 100000,
        resample_every_n_epochs: int = 1,
        resample_batch_size: int = 1,
        bind_init: bool = False,
        depth_scale_shift: int = 0,
        geo_class: bool = False,
        geo_embed_dim: int = 256,
        sae_kwargs: Dict[str, Any] = {},
        criterion_kwargs: Dict[str, Any] = {},
        dead_feature_window: int = 1000,
        name_class: Dict[str, int] = NAME_CLASS,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = self.sae_factory(sae_type, **sae_kwargs)
        self.criterion = criterion_factory(loss_type, **criterion_kwargs)
        self.name_class = name_class

        if "ghost" in loss_type:
            self.use_ghost = True
        else:
            self.use_ghost = False
        self.train_dead_indices = None

        if depth_scale_shift > 0:
            self.geonet = nn.ModuleList()
            self.geonet.append(nn.Linear(4, geo_embed_dim))  # Apply Sin / Cos -> 4
            self.geonet.append(nn.GELU())
            self.geonet.append(nn.LayerNorm(geo_embed_dim))
            for _ in range(depth_scale_shift - 1):
                self.geonet.append(nn.Linear(geo_embed_dim, geo_embed_dim))
                self.geonet.append(nn.GELU())
                self.geonet.append(nn.LayerNorm(geo_embed_dim))

            out_layer = nn.Linear(geo_embed_dim, 2 * self.net.dictionary.in_dimensions)
            # out_layer = nn.Linear(geo_embed_dim, 2)
            nn.init.zeros_(out_layer.bias)
            nn.init.xavier_uniform_(out_layer.weight, gain=0.01)
            self.geonet.append(out_layer)  # scale & shift
            self.geonet = nn.Sequential(*self.geonet)

            if geo_class:
                self.label_embed = nn.Embedding(
                    num_embeddings=2, embedding_dim=geo_embed_dim
                )  # Becareful of the class compare to lat/long encoding
                self.label_mlp = nn.Sequential(
                    nn.Linear(geo_embed_dim, geo_embed_dim),
                    nn.GELU(),
                    nn.Linear(geo_embed_dim, 2 * self.net.dictionary.in_dimensions),  # gate per dimension
                    nn.Sigmoid(),
                )

        else:
            self.geonet = None

        self.geo_class = geo_class
        self.train_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], dead_feature_window)
        self.val_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], None)
        self.test_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], None)

        if bind_init:
            print("Binding the encoder and decoder weights")
            self._initialize_encoder_from_decoder()

        self._val_outputs = []
        self._test_outputs = []
        self.test_r2 = R2Score(num_outputs=1, multioutput="uniform_average")

    @staticmethod
    def sae_factory(sae_type: str, **sae_kwargs) -> nn.Module:

        if sae_type == "topk":
            return TopKSAE(**sae_kwargs)
        elif sae_type == "jump":
            return JumpSAE(**sae_kwargs)
        elif sae_type == "batch_topk":
            return BatchTopKSAE(**sae_kwargs)
        elif sae_type == "vanilla":
            return SAE(**sae_kwargs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def set_arch(self, arch_kwargs: Dict[str, Any] = {}):
        arch_kwargs = OmegaConf.to_container(arch_kwargs, resolve=True)
        points = points_vocab(device=self.device, **arch_kwargs)
        arch_kwargs.pop("csv_path", None)
        arch_kwargs.pop("npy_path", None)
        arch_kwargs.pop("vocab_size", None)
        arch_kwargs.pop("test_batch_size", None)
        arch_kwargs.pop("text_enc_kwargs", None)
        arch_dict = RelaxedArchetypalDictionary(
            in_dimensions=self.net.dictionary.in_dimensions,
            nb_concepts=self.net.nb_concepts,
            points=points,
            device=self.device,
            **arch_kwargs,
        )
        self.net.dictionary = arch_dict

        if self.hparams.bind_init:
            self.net.encoder.final_block[0].weight.copy_(self.net.dictionary.get_dictionary())
            self.net.encoder.final_block[0].bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        lat: Optional[torch.Tensor] = None,
        long: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        flag_params: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:

        if lat is not None and long is not None and self.geonet is not None:
            pos = self._spatial_embedding(lat, long)
            out = self.geonet(pos.squeeze())
            out = out.view(-1, 2, self.net.dictionary.in_dimensions)
            # out = out.view(-1, 2, 1)
            # scale, shift = out[:, 0], out[:, 1]
            if self.geo_class and label is not None:
                label_emb = self.label_embed(label.long()).squeeze()
                gate = self.label_mlp(label_emb)
                gate = gate.view(-1, 2, self.net.dictionary.in_dimensions)
                scale = 1 + torch.tanh(out[:, 0] * gate[:, 0])
                shift = torch.tanh(out[:, 1] * gate[:, 1])
            else:
                scale, shift = 1 + torch.tanh(out[:, 0]), torch.tanh(out[:, 1])
                # scale, shift = 1 + out[:, 0], out[:, 1]
                # scale, shift = 1, torch.tanh(out[:, 1])
                # scale, shift = 1 + torch.tanh(out[:, 0]), 0
            x_mod = scale * x + shift
            z_pre, z, x_hat_mod = self.net(x_mod)
            x_hat = (x_hat_mod - shift) / (scale + 1e-8)  # For metric purposes

            if flag_params:
                return z_pre, z, x_hat, x_mod, scale, shift

            return z_pre, z, x_hat, x_mod

        return self.net(x)

    def _spatial_embedding(self, lat: torch.Tensor, long: torch.Tensor) -> torch.Tensor:
        xpos = torch.concat(
            [
                torch.sin(lat * (math.pi / 180)),
                torch.cos(lat * (math.pi / 180)),
                torch.sin(long * (math.pi / 180)),
                torch.cos(long * (math.pi / 180)),
            ],
            dim=1,
        )
        return xpos

    def step(
        self, batch: Dict[str, Any], tracker: Any, flag_mse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = extract_input(batch)

        if tracker.dead_feature_window is not None:
            ghost_grad_neuron_mask = tracker.n_updates_since_fired > tracker.dead_feature_window
        else:
            ghost_grad_neuron_mask = None

        ### Geolocational Conditioning
        if self.geonet is not None:

            raise NotImplementedError("Geolocational Conditioning is not implemented yet")

            lat = batch["latitude"]
            long = batch["longitude"]
            label = batch["label"]

            ### Different Ways to Forward
            if self.geo_class:
                z_pre, z, x_hat, x_mod, scale, shift = self.forward(x, lat, long, label, flag_params=True)
            else:
                z_pre, z, x_hat, x_mod, scale, shift = self.forward(x, lat, long, flag_params=True)

            ### Different Ways to Compute Loss
            if self.use_ghost:
                loss = self.criterion(
                    x, x_hat, z_pre, z, self.net.get_dictionary(), ghost_grad_neuron_mask, scale=scale, shift=shift
                )
            else:
                loss = self.criterion(x, x_hat, z_pre, z, self.net.get_dictionary(), scale=scale, shift=shift)

        ### No Geolocational Conditioning
        else:
            z_pre, z, x_hat = self.net(x)
            x_mod = None
            if self.use_ghost:
                loss = self.criterion(x, x_hat, z_pre, z, self.net.get_dictionary(), ghost_grad_neuron_mask)
            else:
                loss = self.criterion(x, x_hat, z_pre, z, self.net.get_dictionary())

        tracker.update(z)

        if flag_mse:
            label = batch["label"]
            mse = mse_criterion(x, x_hat, z_pre, z, self.net.get_dictionary())
            return loss, z_pre, z, x, x_hat, x_mod, mse

        return loss, z_pre, z, x, x_hat, x_mod

    def on_train_epoch_start(self):
        if self.current_epoch > 0 and self.current_epoch % self.hparams.resample_every_n_epochs == 0:
            print(f"Epoch {self.current_epoch}: Resampling dead codes...")
            self.net.eval()
            self._resample_dead_codes()
            self.net.train()
            self.log("train/resampled_codes", 1, prog_bar=True)

        print(f"Starting Train Epoch {self.current_epoch} with Dead Ratio {self.train_dead_tracker.get_dead_ratio()}")
        print("Resetting Train Dead Tracker")
        self.train_dead_tracker.alive_features = torch.zeros(
            self.net.get_dictionary().shape[0], dtype=torch.bool, device=self.device
        )

        if self.train_dead_tracker.dead_feature_window is not None:
            ghost_grad_neuron_mask = (
                self.train_dead_tracker.n_updates_since_fired > self.train_dead_tracker.dead_feature_window
            )
            print(f"Ghost Gradient Neuron Mask Sum: {ghost_grad_neuron_mask.sum().item()}")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, codes, inputs, rec_inputs, _ = self.step(batch, tracker=self.train_dead_tracker)
        sparsity_error = l0_eps(codes, 0).sum().item()
        rec_error = _compute_reconstruction_error(inputs, rec_inputs)
        self.log("train/r2", rec_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/l0", sparsity_error, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        dead_ratio = self.train_dead_tracker.get_dead_ratio()
        print(f"Dead Train Ratio {dead_ratio}")
        self.log("train/dead_features", dead_ratio, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_dead_tracker.alive_features = torch.zeros(
            self.net.get_dictionary().shape[0], dtype=torch.bool, device=self.device
        )

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, codes, inputs, rec_inputs, _, mse = self.step(
            batch, tracker=self.val_dead_tracker, flag_mse=True
        )

        # Compute codes mean for class specificty error
        assert (
            len(batch["label"].shape) == 1 or batch["label"].shape[1] == 1
        ), "Class specific error need labels in shape N*1 or N"
        if len(batch["label"].shape) > 1:
            label = batch["label"].squeeze()
        else:
            label = batch["label"]
        codes_stats = {}
        for class_id in torch.unique(label):
            class_codes = codes[label == class_id, :]
            class_codes_sum = class_codes.sum(dim=0)
            class_codes_count = (class_codes > 0).sum(dim=0)
            class_codes_count_tot = class_codes.shape[0]
            codes_stats[int(class_id.item())] = (
                class_codes_sum.detach().cpu(),
                class_codes_count.detach().cpu(),
                class_codes_count_tot,
            )

        sparsity_error = l0_eps(codes, 0).sum().item()
        hoyer_error = hoyer(codes).mean().item()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l0", sparsity_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l2", avg_l2_loss(inputs, rec_inputs), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hoyer", hoyer_error, on_step=False, on_epoch=True, prog_bar=True)
        self._val_outputs.append(
            {
                "loss": loss,
                "inputs": inputs.detach().cpu(),
                "rec_inputs": rec_inputs.detach().cpu(),
                # "label": batch["label"].detach().cpu(),
                "codes_stats": codes_stats,
            }
        )
        return {"loss": loss}

    # Potentially Add Frechet & Wasserstein
    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        inputs = torch.cat([x["inputs"] for x in outputs], dim=0)
        rec_inputs = torch.cat([x["rec_inputs"] for x in outputs], dim=0)
        # label = torch.cat([x["label"] for x in outputs], dim=0)

        alive_features = self.val_dead_tracker.alive_features
        selectivity, selectivity_firing, class_size, class_size_firing = self.class_selectivity(
            codes_stats=[x["codes_stats"] for x in outputs], alive_features=alive_features.cpu()
        )
        self._val_outputs.clear()
        rec_error = _compute_reconstruction_error(inputs, rec_inputs)
        self.log("val/r2", rec_error, prog_bar=True)
        self.log("val/class_selectivity", selectivity)
        self.log("val/class_selectivity_firing", selectivity_firing)

        for class_id, size in class_size.items():
            self.log(f"val/class_size_{self.name_class[class_id]}", size, prog_bar=False)

        for class_id, size in class_size_firing.items():
            self.log(f"val/class_size_firing_{self.name_class[class_id]}", size, prog_bar=False)

        # Computing Val Dead Ratio
        dead_ratio = self.val_dead_tracker.get_dead_ratio()
        print(f"Dead Val Ratio {dead_ratio}")
        self.log("val/dead_features", dead_ratio, prog_bar=True)

        # Dict Measures
        stable_rank = compute_stable_rank(self.net.get_dictionary()[alive_features].detach())
        eff_rank = compute_effective_rank(self.net.get_dictionary()[alive_features].detach())
        coherence = compute_coherence(self.net.get_dictionary()[alive_features].detach())
        self.log("val/stable_rank", stable_rank)
        self.log("val/eff_rank", eff_rank)
        self.log("val/coherence", coherence)

    def on_test_epoch_start(self):
        self.test_dead_tracker.alive_features = torch.zeros(
            self.net.get_dictionary().shape[0], dtype=torch.bool, device=self.device
        )
        self.test_r2.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, codes, inputs, rec_inputs, _, mse = self.step(
            batch, tracker=self.test_dead_tracker, flag_mse=True
        )

        # Compute codes mean for class specificty error
        assert (
            len(batch["label"].shape) == 1 or batch["label"].shape[1] == 1
        ), "Class specific error need labels in shape N*1 or N"
        if len(batch["label"].shape) > 1:
            label = batch["label"].squeeze()
        else:
            label = batch["label"]
        codes_stats = {}
        for class_id in torch.unique(label):
            class_codes = codes[label == class_id, :]
            class_codes_sum = class_codes.sum(dim=0)
            class_codes_count = (class_codes > 0).sum(dim=0)
            class_codes_count_tot = class_codes.shape[0]
            codes_stats[int(class_id.item())] = (
                class_codes_sum.detach().cpu(),
                class_codes_count.detach().cpu(),
                class_codes_count_tot,
            )

        sparsity_error = l0_eps(codes, 0).sum().item()
        hoyer_error = hoyer(codes).mean().item()
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/l0", sparsity_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/l2", avg_l2_loss(inputs, rec_inputs), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hoyer", hoyer_error, on_step=False, on_epoch=True, prog_bar=True)
        self._update_r2(self.test_r2, inputs, rec_inputs)
        self._test_outputs.append(
            {
                "loss": loss,
                # "inputs": inputs.detach().cpu(),
                # "rec_inputs": rec_inputs.detach().cpu(),
                # "label": batch["label"].detach().cpu(),
                # "codes": codes.detach().cpu(),
                "codes_stats": codes_stats,
            }
        )
        return {"loss": loss}

    def _update_r2(self, metric, x, x_hat):
        x, x_hat = x.detach().cpu(), x_hat.detach().cpu()
        if len(x.shape) == 4 and len(x_hat.shape) == 2:
            x = rearrange(x, 'n c w h -> (n w h) c')
        elif len(x.shape) == 3 and len(x_hat.shape) == 2:
            x = rearrange(x, 'n t c -> (n t) c')
        else:
            assert x.shape == x_hat.shape, "Input and output shapes must match."
        metric.update(x_hat.reshape(-1), x.reshape(-1))  # preds, target — both flattened to 1D

    # Potentially Add Frechet & Wasserstein
    def on_test_epoch_end(self):
        outputs = self._test_outputs
        # inputs = torch.cat([x["inputs"] for x in outputs], dim=0)
        # rec_inputs = torch.cat([x["rec_inputs"] for x in outputs], dim=0)
        # label = torch.cat([x["label"] for x in outputs], dim=0)

        alive_features = self.test_dead_tracker.alive_features
        selectivity, selectivity_firing, class_size, class_size_firing = self.class_selectivity(
            codes_stats=[x["codes_stats"] for x in outputs], alive_features=alive_features.cpu()
        )
        self._test_outputs.clear()
        #rec_error = _compute_reconstruction_error(inputs, rec_inputs)
        # self.log("test/r2", rec_error, prog_bar=True)
        rec_error = self.test_r2.compute().item()
        self.log("test/r2", rec_error, prog_bar=True)
        self.log("test/class_selectivity", selectivity)
        self.log("test/class_selectivity_firing", selectivity_firing)

        for class_id, size in class_size.items():
            self.log(f"test/class_size_{self.name_class[class_id]}", size, prog_bar=False)
            self.log(f"test/class_size_firing_{self.name_class[class_id]}", class_size_firing[class_id], prog_bar=False)

        # Computing Test Dead Ratio
        dead_ratio = self.test_dead_tracker.get_dead_ratio()
        print(f"Dead Test Ratio {dead_ratio}")
        self.log("test/dead_features", dead_ratio, prog_bar=True)

        # Dict Measures
        stable_rank = compute_stable_rank(self.net.get_dictionary()[alive_features].detach())
        eff_rank = compute_effective_rank(self.net.get_dictionary()[alive_features].detach())
        coherence = compute_coherence(self.net.get_dictionary()[alive_features].detach())
        self.log("test/stable_rank", stable_rank)
        self.log("test/eff_rank", eff_rank)
        self.log("test/coherence", coherence)

    @torch.no_grad()
    def _resample_dead_codes(self):
        """Implements resampling of dead codes from
        https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling"""

        dead_indices = torch.where(self.train_dead_tracker.alive_features == False)[0]
        if len(dead_indices) == 0:
            return

        dataset = self.trainer.train_dataloader.dataset

        sampled_indices = random.sample(range(len(dataset)), self.hparams.num_samples)
        subset = Subset(dataset, sampled_indices)
        subset_dataloader = DataLoader(subset, batch_size=self.hparams.resample_batch_size, shuffle=False)

        mse_loss = criterion_factory(loss_type="mse", aggregate_batch=False)

        tot_indices = []
        tot_loss = []
        for i, batch in enumerate(subset_dataloader):
            x = extract_input(batch)
            x = x.to(self.device)

            if self.geonet is not None:
                raise NotImplementedError("Geolocational Conditioning is not implemented yet")
                lat = batch["latitude"].to(self.device)
                long = batch["longitude"].to(self.device)
                label = batch["label"].to(self.device)
                if self.geo_class:
                    z_pre, z, x_hat, _ = self.forward(x, lat, long, label)
                else:
                    z_pre, z, x_hat, _ = self.forward(x, lat, long)
            else:
                z_pre, z, x_hat = self.net(x)

            loss = mse_loss(x, x_hat, z_pre, z, self.net.get_dictionary())
            start = i * self.hparams.resample_batch_size
            end = start + loss.shape[0]

            tot_indices.extend(sampled_indices[start:end])
            tot_loss.append(loss.pow(2).detach().cpu())

        tot_loss = torch.cat(tot_loss, dim=0)
        tot_probs = tot_loss / tot_loss.sum()

        chosen_indices = torch.multinomial(tot_probs, num_samples=len(dead_indices), replacement=False)
        chosen_dataset_indices = [tot_indices[i] for i in chosen_indices]
        for dead_idx, chosen_idx in zip(dead_indices, chosen_dataset_indices):
            raw_input = dataset[chosen_idx]
            sampled_input = extract_input(raw_input).to(self.device)

            # RAW Resample
            if self.geonet is not None:
                raise NotImplementedError("Geolocational Conditioning is not implemented yet")
                lat = raw_input["latitude"].to(self.device)
                long = raw_input["longitude"].to(self.device)
                label = raw_input["label"].to(self.device)
                if self.geo_class:
                    _, _, _, sampled_input = self.forward(
                        sampled_input.unsqueeze(0), lat.unsqueeze(0), long.unsqueeze(0), label.unsqueeze(0)
                    )
                else:
                    _, _, _, sampled_input = self.forward(
                        sampled_input.unsqueeze(0), lat.unsqueeze(0), long.unsqueeze(0)
                    )

            sampled_input = sampled_input / sampled_input.norm(p=2)
            self.net.dictionary._weights[dead_idx, :] = sampled_input  # Based on Overcomplete Framework
            alive_norm = (
                self.net.encoder.final_block[0].weight[self.train_dead_tracker.alive_features, :].norm(dim=1)
            )  # Dimension should be n_concept, last_dimension
            mean_alive_norm = alive_norm.mean()
            target_norm = mean_alive_norm * 0.2

            self.net.encoder.final_block[0].weight[dead_idx, :] = sampled_input * target_norm
            self.net.encoder.final_block[0].bias[dead_idx] = 0.0

            optimizer = self.optimizers()
            if isinstance(optimizer, torch.optim.Adam):
                # Handle encoder weight row
                enc_weight_param = self.net.encoder.final_block[0].weight
                enc_bias_param = self.net.encoder.final_block[0].bias
                dec_weight_param = self.net.dictionary._weights
                for param, index in [
                    (enc_weight_param, dead_idx),
                    (enc_bias_param, dead_idx),
                    (dec_weight_param, dead_idx),
                ]:
                    state = optimizer.state.get(param, {})
                    if state:
                        state = optimizer.state[param]
                        if "exp_avg" in state and "exp_avg_sq" in state:
                            state["exp_avg"][index].zero_()
                            state["exp_avg_sq"][index].zero_()
            else:
                raise NotImplementedError("Specify reset for other optimizer types")

    @torch.no_grad()
    def _initialize_encoder_from_decoder(self):
        """Set encoder final layer weights equal to decoder dictionary weights."""
        self.net.encoder.final_block[0].weight.copy_(self.net.dictionary._weights)
        self.net.encoder.final_block[0].bias.zero_()

    @torch.no_grad()
    def set_init_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pos_class_budget: float = 0.5,
        mode: str = "kmeans",
        neg_max_samples: int = 2e5,
    ):
        tot_size = self.net.nb_concepts
        pos_tot_size = int(tot_size * pos_class_budget)
        pos_idx = np.random.randint(0, tot_size, pos_tot_size)
        neg_idx = np.array([idx for idx in range(tot_size) if idx not in pos_idx])

        X_pos = X[y == 1]
        X_neg = X[y == 0]

        print(X_pos.shape, X_neg.shape)
        if mode in ["kmeans", "kmeans-under"]:
            print("KMeans Positive Initialization")
            kmean = KMeans(n_clusters=pos_tot_size)
            kmean.fit(X_pos)
            pos_codes_init = kmean.cluster_centers_
            pos_norms = np.linalg.norm(pos_codes_init, axis=1, keepdims=True)
            pos_codes_init = pos_codes_init / pos_norms
            if mode == "kmeans-under":
                assert (
                    neg_max_samples < X_neg.shape[0]
                ), "Negative samples should be less than available samples for under-sampling"
                assert neg_max_samples >= pos_tot_size, "Negative samples should be more than positive codes for kmeans"
                X_neg = X_neg[np.random.choice(X_neg.shape[0], size=int(neg_max_samples), replace=False)]

            print("KMeans Negative Initialization")
            kmean = KMeans(n_clusters=neg_idx.shape[0])
            kmean.fit(X_neg)
            neg_codes_init = kmean.cluster_centers_
            neg_norms = np.linalg.norm(neg_codes_init, axis=1, keepdims=True)
            neg_codes_init = neg_codes_init / neg_norms
        elif mode == "random":
            pos_codes_init = X_pos[np.random.choice(X_pos.shape, size=pos_tot_size, replace=False)]
            neg_codes_init = X_pos[np.random.choice(X_neg.shape, size=neg_idx.shape[0], replace=False)]

        else:
            raise NotImplementedError("Class specific initialization ")

        pos_norms = np.linalg.norm(pos_codes_init, axis=1, keepdims=True)
        pos_codes_init = pos_codes_init / pos_norms
        neg_norms = np.linalg.norm(neg_codes_init, axis=1, keepdims=True)
        neg_codes_init = neg_codes_init / neg_norms

        device = self.net.dictionary._weights.device
        dtype = self.net.dictionary._weights.dtype
        pos_idx = torch.from_numpy(pos_idx).to(device).long()
        neg_idx = torch.from_numpy(neg_idx).to(device).long()
        pos_codes_init = torch.from_numpy(pos_codes_init).to(device).type(dtype)
        neg_codes_init = torch.from_numpy(neg_codes_init).to(device).type(dtype)

        self.net.dictionary._weights[pos_idx] = pos_codes_init
        self.net.dictionary._weights[neg_idx] = neg_codes_init
        if self.hparams.bind_init:
            self._initialize_encoder_from_decoder()

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = optimizer_factory(
            optim_type=self.hparams.optimizer_type,
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = scheduler_factory(scheduler_type=self.hparams.scheduler_type, optimizer=optimizer)

        if lr_scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss"}

    @staticmethod
    def class_selectivity(
        codes_stats: List[Dict[int, List]], alive_features: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
        tot_codes_sum = {}
        tot_codes_count = {}
        tot_codes_count_tot = {}
        for stats in codes_stats:
            for class_id, (codes_sum, codes_count, codes_count_tot) in stats.items():
                if class_id not in tot_codes_sum:
                    tot_codes_sum[class_id] = codes_sum
                    tot_codes_count[class_id] = codes_count
                    tot_codes_count_tot[class_id] = codes_count_tot
                else:
                    tot_codes_sum[class_id] += codes_sum
                    tot_codes_count[class_id] += codes_count
                    tot_codes_count_tot[class_id] += codes_count_tot

        class_codes_mean = torch.stack(
            [tot_codes_sum[class_id] / (tot_codes_count[class_id] + 1e-12) for class_id in tot_codes_sum], dim=1
        )
        class_codes_firing_rate = torch.stack(
            [tot_codes_count[class_id] / tot_codes_count_tot[class_id] for class_id in tot_codes_count], dim=1
        )

        class_codes_mean = class_codes_mean[alive_features, :] if alive_features is not None else class_codes_mean
        class_codes_firing_rate = (
            class_codes_firing_rate[alive_features, :] if alive_features is not None else class_codes_firing_rate
        )

        if (class_codes_mean.shape[1] < 2) or (class_codes_firing_rate.shape[1] < 2):
            return 0.0, 0.0, {}, {}

        max_codes, max_idx = class_codes_mean.max(dim=1)
        values, counts = torch.unique(max_idx, return_counts=True)
        class_size = {v.item(): c.item() for v, c in zip(values, counts / max_idx.shape[0])}

        sum_codes = class_codes_mean.sum(dim=1)
        other_means = (sum_codes - max_codes) / (class_codes_mean.shape[1] - 1)
        selectivity_per_code = (max_codes - other_means) / (max_codes + other_means + 1e-12)

        max_firing, max_idx = class_codes_firing_rate.max(dim=1)
        values, counts = torch.unique(max_idx, return_counts=True)
        class_size_firing = {v.item(): c.item() for v, c in zip(values, counts / max_idx.shape[0])}

        sum_codes_firing = class_codes_firing_rate.sum(dim=1)
        other_means_firing = (sum_codes_firing - max_firing) / (class_codes_firing_rate.shape[1] - 1)
        selectivity_per_code_firing = (max_firing - other_means_firing) / (max_firing + other_means_firing + 1e-12)

        return (
            selectivity_per_code.mean().item(),
            selectivity_per_code_firing.mean().item(),
            class_size,
            class_size_firing,
        )