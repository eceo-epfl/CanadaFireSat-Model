import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from omegaconf import OmegaConf
import numpy as np
import os
from pathlib import Path

#from src.models.l1c2l2a_adapter import L1C2L2AAdapter
#from src.CBM.concepts_minimal import _load_sae_from_ckpt

from experiments.concept_bottleneck.models.clearclip import maybe_patch_clearclip
from experiments.concept_bottleneck.models.vpt import VPTAdapter
from msclip.inference.utils import build_model
#from msclip.inference.clearclip import maybe_patch_clearclip
#from msclip.inference.sclip import maybe_patch_sclip

#from src.models.sae import plSAE
#from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary


class DOYEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(2, embed_dim)

    def forward(self, doy):
        # doy: [B, T] integers (0–1)
        theta = 2 * math.pi * doy
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        cyc = torch.stack([sin, cos], dim=-1)   # [B, T, 2]
        return self.fc(cyc)                     # [B, T, D]


class TemporalConvMixer(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=2.0, dropout=0.1, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        #[B*P, D, T] -> [B*P, D, T] input size should be equal to ouptut size
        self.temporal_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
            bias=False,
        )
        self.temporal_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, doy_emb=None, mask=None):

        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()  # [B*P, T, 1]
            x = x * valid                          
            if doy_emb is not None:
                doy_emb = doy_emb * valid          

        y = x + (doy_emb if doy_emb is not None else 0.0)
        y = self.norm1(y)                 # [B*P, T, D]
        y = y.transpose(1, 2)             
        y = self.temporal_conv(y)
        y = self.temporal_dropout(y)
        y = y.transpose(1, 2)            

        x = x + y                         

        x = x + self.mlp(self.norm2(x))

        if mask is not None:
            x = x * valid                

        return x                          # [B*P, T, D]


class TemporalMixer(nn.Module):
    """
    Small, stable pre-projection temporal mixer (keeps sequence length T).
    Pre-norm MHA + MLP with residuals; DOY can nudge Q/K but leaves values on the identity path.
    """
    def __init__(self, embed_dim=768, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.last_attn = None  # [B*P, T, T]
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    # TODO: Add potentially th extra-masking
    def forward(self, x, doy_emb=None, mask=None):
        # x: [B*P, T, 768], doy_emb: [B*P, T, 768] or None
        qk = self.norm1(x + (doy_emb if doy_emb is not None else 0.0))
        y, attn = self.attn(qk, qk, x, need_weights=True, average_attn_weights=True,key_padding_mask= mask)
        self.last_attn = attn.detach()  # [B*P, T, T]
        x = x + y                              # residual
        x = x + self.mlp(self.norm2(x))        # FFN block
        return x                                # [B*P, T, 768]


class MSClipTemporalCBM(nn.Module):
    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        patch_size: int = 16, # TODO: Should be obtained from msclip
        channels=10,
        num_classes=2,
        out_H=25,
        out_W=25,
        freeze_msclip=True, #TODO: Becareful it seems not used
        use_doy=True,
        ds_labels=True,
        use_cls_fusion=False,
        img_res: int = 224,
        upscale_factor: int = 1,
        # use_l1c2l2a_adapter: bool = False,
        # log_concepts: bool = False,
        # l1c2l2a_dropout: int = 0,
        # l1c2l2a_Adapter_loc:str = "",
        # learned_query: bool = False,
        use_mixer: bool = True, # Used when outputting the features
        # use_CBM: bool = False, # Also used in eval and features
        # sae_config: Optional[str] = None,
        pretrained: bool = True,
        clearclip: Dict[str, Any] = None,
        vpt_config: Dict[str, Any] = None,
        # sclip: Dict[str, Any] = None,
        # denseclip: Dict[str, Any] = None,
        # sae_before_attention: bool = False,
        # concept_attn_temperature: float = 1.0,
        # sae_encode_chunk_size: int = 2048,
        use_ln_norm_patch: bool = False,
        **kwargs,
        ):
        super().__init__()

        print("### INITIALIZING MSCLIP MODEL ###")
        self.ds_labels = ds_labels
        self.out_H = out_H
        self.out_W = out_W
        self.channels = channels
        self.use_doy = use_doy
        self.image_size = img_res
        self.patch_size = patch_size
        self.use_cls_fusion = use_cls_fusion # TODO: Not use and could be interesting
        self.use_mixer = use_mixer
        self.use_ln_norm_patch = use_ln_norm_patch
        # self.log_concepts = log_concepts
        # self.sae_before_attention = bool(sae_before_attention)
        # self.concept_attn_temperature = float(concept_attn_temperature)
        #self.sae_encode_chunk_size = int(sae_encode_chunk_size)
        self.last_time_attn = None  # [B, P, T] when sae_before_attention=True
        self.concept_time_query = None  # nn.Parameter set after SAE is loaded
        self.editing_vector = None


        msclip_model, _, tokenizer = build_model(
            model_name=model_name, pretrained=pretrained, ckpt_path=ckpt_path, device="cpu", channels=channels
        )

        # Interpolate positional embeddings if img_res differs from default (224)
        if img_res != 224:
            self._interpolate_pos_embed(msclip_model, img_res, patch_size)

        self.msclip_model = msclip_model

        # -- ClearCLIP
        if clearclip is not None and clearclip["enabled"]:
            num_patched = maybe_patch_clearclip(self.msclip_model.image_encoder, clearclip)
            print("Patched clearclip : ", num_patched)
            if num_patched > 0:
                print(f"[ClearCLIP] Patched last {num_patched} vision blocks "
                    f"(keep_ffn={clearclip.get('keep_ffn', False)}, "
                    f"keep_residual={clearclip.get('keep_residual', False)})")

        self.tokenizer = tokenizer
        self.vision = self.msclip_model.clip_base_model.model.visual
        self.vision.output_tokens = True

        """
        # --- SCLIP
        if sclip is not None and sclip["enabled"]:
            num_patched = maybe_patch_sclip(self.image_encoder, sclip)
            if num_patched > 0:
                print(f"[SCLIP] Patched last {num_patched} vision blocks "
                      f"(CSA attention)")
        """

        self.embed_dim = 512 # TODO: Should be obtained from msclip
        self.mix_dim   = 768 # TODO: Should be obtained from msclip
        self.H_patch = self.image_size // self.patch_size
        self.W_patch = self.image_size // self.patch_size
        self.num_patches = self.H_patch * self.W_patch
        self.has_cls_token = True # Only used in eval
        self.upscale_factor = upscale_factor

        #self.use_l1c2l2a_adapter = use_l1c2l2a_adapter
        #self.l1c2l2a_dropout = l1c2l2a_dropout

        """
        if self.use_l1c2l2a_adapter:  #Test
            self.l1c2l2a = L1C2L2AAdapter(dim=self.embed_dim, dropout=self.l1c2l2a_dropout)
            adapter_weights = torch.load("/home/grosse/wildfire-forecast/worldstrat/l1c2l2a_linear.pt", map_location="cpu")
            self.l1c2l2a.load_state_dict(adapter_weights)"""

        if self.use_doy:
            self.doy_embed_mix  = DOYEmbed(self.mix_dim)
            self.doy_embed_pool = DOYEmbed(self.embed_dim) # Why we need this during inference

        self.temporal_mixer = TemporalMixer(embed_dim=self.mix_dim, num_heads=4, mlp_ratio=2.0, dropout=0.1) # should not be hardcoded

        # self.use_CBM = use_CBM
        """
        self.last_concept_map = None
        self.last_concept_map_raw = None
        # When sae_before_attention=True and log_concepts=True, we store three concept maps
        self.last_concept_map_last = None
        self.last_concept_map_mean = None
        self.last_concept_map_delta = None
        self.last_concept_map_last_raw = None
        self.last_concept_map_mean_raw = None
        self.last_concept_map_delta_raw = None"""

        """
        if self.use_CBM:
            cfg_sae = OmegaConf.load(sae_config)
            sae_model_config = OmegaConf.to_container(cfg_sae["sae"], resolve=True)
            sae_model_config.pop("_target_", None)

            if cfg_sae["use_archetypal"]["enabled"]:
                points = torch.tensor(np.load(Path(cfg_sae["sae_ckpt_path"]).parent / "archetypalPoints.npy"))
            else:
                points = None

            self.sae = plSAE(points=points, **sae_model_config)

            ckpt = torch.load(cfg_sae["sae_ckpt_path"], map_location="cuda:0")
            state_dict = ckpt["state_dict"]

            if any(k.startswith("msclip_model.") for k in state_dict.keys()):
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("msclip_model.")}

            incompat = self.sae.load_state_dict(state_dict, strict=False)

            missing, unexpected = self.sae.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                print("SAE load_state_dict — missing:", missing, "unexpected:", unexpected)

            self.sae.eval().to("cuda:0")

            for p in self.sae.net.parameters():
                p.requires_grad = False

            concept_dim = int(cfg_sae["nb_concepts"])
            self.concept_dim = concept_dim

            # In sae_before_attention mode we append cyclic DOY as two channels (sin, cos) -> (C+2) = 8194
            # and use the Option-B temporal summary: [last, mean, delta] -> 3*(C+2) channels.
            if self.sae_before_attention:
                self.concept_dim_plus = concept_dim + 2
                self.head = nn.Conv2d(3 * self.concept_dim_plus, num_classes, 1)
            else:
                self.head = nn.Conv2d(concept_dim, num_classes, 1)


        else:
            self.head = nn.Conv2d(self.embed_dim, num_classes, 1)
            nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)"""

        self.head = nn.Conv2d(self.embed_dim, num_classes * self.upscale_factor, 1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)


        for p in self.msclip_model.parameters():
            p.requires_grad = False


        # Attention HAS TO BE After freezing msclip
        if vpt_config is not None and vpt_config.get("enabled", False):
            self.vpt = VPTAdapter(
                vision=self.vision,
                num_tokens=vpt_config.get("num_tokens", 20),
                total_d_layer=vpt_config.get("total_d_layer", 11),
                prompt_dim=self.mix_dim,
                dropout=vpt_config.get("dropout", 0.1),
            )
        else:
            self.vpt = None

    def _interpolate_pos_embed(self, msclip_model, img_res: int, patch_size: int):
        vision = msclip_model.clip_base_model.model.visual
        pos_embed = vision.positional_embedding  # (1 + old_num_patches, dim)

        cls_token  = pos_embed[:1, :]   # (1, dim)
        patch_embed = pos_embed[1:, :]  # (old_num_patches, dim)

        old_size = int(patch_embed.shape[0] ** 0.5)  # e.g. 14 for 224/16
        new_size = img_res // patch_size              # e.g. 8 for 128/16

        if old_size != new_size:
            print(f"Interpolating positional embeddings: {old_size}x{old_size} -> {new_size}x{new_size}")
            dim = patch_embed.shape[1]
            patch_embed = patch_embed.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)  # (1, dim, old, old)
            patch_embed = F.interpolate(patch_embed, size=(new_size, new_size), mode="bicubic", align_corners=False)
            patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(new_size * new_size, dim)    # (new_num_patches, dim)

            new_pos_embed = torch.cat([cls_token, patch_embed], dim=0)
            vision.positional_embedding = nn.Parameter(new_pos_embed)
            print(f"positional_embedding resized: {pos_embed.shape} -> {new_pos_embed.shape}")


    def forward(self, batch, doy=None, seq_len=None):
        # [B, T, C, H, W]
        assert batch.ndim == 5, f"inputs must be [B,T,C,H,W], got {batch.ndim} dims"
        B, T, C, H, W = batch.shape

        #print(f"Input batch shape: {batch.shape}")
        #print(torch.amin(batch, dim=(0, 1, 3, 4)), torch.amax(batch, dim=(0, 1, 3, 4)), torch.mean(batch, dim=(0, 1, 3, 4)), torch.std(batch, dim=(0, 1, 3, 4)))
        #print(f"DOY shape: {doy.shape if doy is not None else None}")
        #print(f"DOY Min: {torch.amin(doy, dim=(2, 3, 4)) if doy is not None else None}, Max: {torch.amax(doy, dim=(2, 3, 4)) if doy is not None else None}")

        if seq_len is None:
            # If caller doesn't provide true sequence lengths, assume all timesteps are valid.
            seq_len = torch.full((B,), T, device=batch.device, dtype=torch.long)

        assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"

        x = batch.reshape(B * T, C, H, W)

        t_idx = torch.arange(T, device=batch.device).unsqueeze(0)      # [1,T]
        valid_BT  = t_idx < seq_len.unsqueeze(1)                       # [B,T] True=valid
        P = self.num_patches
        valid_BPT = valid_BT.unsqueeze(1).expand(-1, P, -1)            # [B,P,T]
        valid_mask = valid_BPT.reshape(B * P, T)                       # [B*P,T]

        if self.vpt is not None:
            # Run VPT forward manually through the vision encoder
            B_T = x.shape[0]
            v = self.vision


            # Replicate VisionTransformer.forward up to transformer blocks
            feat = v.conv1(x)                                              # [B*T, D, H_p, W_p]
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)         # [B*T, D, P]
            feat = feat.permute(0, 2, 1)                                   # [B*T, P, D]
            feat = torch.cat([
                v.class_embedding.unsqueeze(0).unsqueeze(0).expand(B_T, -1, -1),
                feat
            ], dim=1)                                                      # [B*T, 1+P, D]
            feat = feat + v.positional_embedding.unsqueeze(0)
            feat = v.ln_pre(feat)
            # feat = feat.permute(1, 0, 2)                                   # LND

            feat = self.vpt(feat)                                          # [1+P, B*T, D] LND

            #feat = feat.permute(1, 0, 2)                                   # NLD [B*T, 1+P, D]
            pooled_feats = feat[:, 0]                                      # CLS [B*T, D]
            patch_feats  = feat[:, 1:]                                     # patches [B*T, P, D]

            # Apply ln_post and proj as normal
            # patch_feats = v.ln_post(patch_feats) @ v.proj                 # [B*T, P, 512]
            # Undo proj for mix_dim path (patch_feats before proj needed by temporal mixer)
            # So instead keep pre-proj features:
            pooled_feats    = v.ln_post(pooled_feats) if self.use_ln_norm_patch else pooled_feats
            pooled_feats = pooled_feats @ v.proj            # [B*T, 512]
        else:
            pooled_feats, patch_feats = self.msclip_model.image_encoder(x)

        """
        if self.use_CBM and self.sae_before_attention:

            patch_512 = self.vision.ln_post(patch_feats)              # [B*T, P, 768]
            patch_512 = patch_512 @ self.vision.proj                 # [B*T, P, 512]
            patch_512 = patch_512.view(B, T, self.num_patches, self.embed_dim)  # [B, T, P, 512]

            # 2) Encode ALL (time,patch) tokens with the frozen SAE (no chunking)
            tokens = patch_512.reshape(-1, self.embed_dim)            # [B*T*P, 512]
            with torch.no_grad():
                z_pre, z0 = self.sae.net.encode(tokens.float())        # z0: [B*T*P, C]

            # Apply concept ablation gate AFTER logging raw concepts (pre-gate).
            if self.editing_vector is not None:
                gate = self.editing_vector.to(z0.device).view(1, -1)
                z = z0 * gate
            else:
                z = z0

            # 3) Reshape to concept maps per timestep: Z_raw/Z = [B, T, C, H_p, W_p]
            Z_raw = z0.view(B, T, self.num_patches, -1)
            Z_raw = Z_raw.view(B, T, self.H_patch, self.W_patch, -1).permute(0, 1, 4, 2, 3).contiguous()

            Z = z.view(B, T, self.num_patches, -1)
            Z = Z.view(B, T, self.H_patch, self.W_patch, -1).permute(0, 1, 4, 2, 3).contiguous()

            if doy is not None:
                if doy.ndim > 2:
                    doy_use = doy.view(B, T, -1)[:, :, 0]
                else:
                    doy_use = doy
                # doy is expected to already be in [0, 1] (fraction of the year)
                theta = 2.0 * math.pi * doy_use.to(Z.dtype)
                sin = torch.sin(theta)
                cos = torch.cos(theta)
                cyc = torch.stack([sin, cos], dim=2)  # [B, T, 2]
                doy_chan = cyc.unsqueeze(3).unsqueeze(4)  # [B, T, 2, 1, 1]
                doy_chan = doy_chan.expand(-1, -1, 2, self.H_patch, self.W_patch)  # [B, T, 2, H_p, W_p]
            else:
                doy_chan = torch.zeros((B, T, 2, self.H_patch, self.W_patch), device=Z.device, dtype=Z.dtype)

            # Append DOY channels to both raw and gated concept tensors
            Z_raw = torch.cat([Z_raw, doy_chan], dim=2)  # [B, T, C+2, H_p, W_p]
            Z     = torch.cat([Z,     doy_chan], dim=2)  # [B, T, C+2, H_p, W_p]

            # 5) Mask padded timesteps
            M = valid_BT.to(Z.dtype).view(B, T, 1, 1, 1)  # [B,T,1,1,1]
            Z_raw = Z_raw * M
            Z     = Z * M

            den = M.sum(dim=1).clamp_min(1.0)                              # [B,1,1,1]
            Z_mean     = Z.sum(dim=1) / den                                # [B,C+2,H_p,W_p]
            Z_mean_raw = Z_raw.sum(dim=1) / den                            # [B,C+2,H_p,W_p]

            last_idx = (seq_len - 1).clamp_min(0)                          # [B]
            prev_idx = (last_idx - 1).clamp_min(0)                         # [B]
            b_idx = torch.arange(B, device=Z.device)

            Z_last     = Z[b_idx, last_idx]                                # [B,C+2,H_p,W_p]
            Z_prev     = Z[b_idx, prev_idx]                                # [B,C+2,H_p,W_p]
            Z_delta    = Z_last - Z_prev                                   # [B,C+2,H_p,W_p]

            Z_last_raw  = Z_raw[b_idx, last_idx]                           # [B,C+2,H_p,W_p]
            Z_prev_raw  = Z_raw[b_idx, prev_idx]                           # [B,C+2,H_p,W_p]
            Z_delta_raw = Z_last_raw - Z_prev_raw                          # [B,C+2,H_p,W_p]

            # Final features: [B, 3*(C+2), H_p, W_p]
            feats = torch.cat([Z_last, Z_mean, Z_delta], dim=1)

            if self.log_concepts:
                # Store per-summary concept maps (exclude DOY channels) for evaluation / logging.
                n_extra = int(Z_last.shape[1] - getattr(self, 'concept_dim', Z_last.shape[1]))
                n_extra = max(n_extra, 0)

                if n_extra > 0:
                    z_last_log      = Z_last[:, :-n_extra].detach()
                    z_mean_log      = Z_mean[:, :-n_extra].detach()
                    z_delta_log     = Z_delta[:, :-n_extra].detach()

                    z_last_raw_log  = Z_last_raw[:, :-n_extra].detach()
                    z_mean_raw_log  = Z_mean_raw[:, :-n_extra].detach()
                    z_delta_raw_log = Z_delta_raw[:, :-n_extra].detach()
                else:
                    z_last_log      = Z_last.detach()
                    z_mean_log      = Z_mean.detach()
                    z_delta_log     = Z_delta.detach()

                    z_last_raw_log  = Z_last_raw.detach()
                    z_mean_raw_log  = Z_mean_raw.detach()
                    z_delta_raw_log = Z_delta_raw.detach()

                # RAW = pre-ablation (pre editing_vector)
                self.last_concept_map_last_raw  = z_last_raw_log.clone()
                self.last_concept_map_mean_raw  = z_mean_raw_log.clone()
                self.last_concept_map_delta_raw = z_delta_raw_log.clone()

                # (Optionally) also expose the gated versions actually used for prediction
                self.last_concept_map_last  = z_last_log
                self.last_concept_map_mean  = z_mean_log
                self.last_concept_map_delta = z_delta_log

                # Backward-compatible alias: last timestep summary
                self.last_concept_map_raw = self.last_concept_map_last_raw
                self.last_concept_map     = self.last_concept_map_last

            logits_map = self.head(feats)  # [B, num_classes, H_p, W_p]
            out = F.interpolate(
                logits_map, size=(H, W) if not self.ds_labels else (self.out_H, self.out_W),
                mode='bilinear', align_corners=False
            )
            return out"""

        patch_feats = patch_feats.view(B, T, self.num_patches, self.mix_dim) \
                                .permute(0, 2, 1, 3).contiguous() \
                                .view(B * self.num_patches, T, self.mix_dim)          # [B*P, T, 768]

        # DOY for mixer (768-d)
        doy_mix = None
        if self.use_doy and (doy is not None):
            if doy.ndim > 2:
                doy = doy.view(B, T, -1)[:, :, 0]
            assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
            d = self.doy_embed_mix(doy).unsqueeze(1).expand(-1, self.num_patches, -1, -1) \
                                    .reshape(B * self.num_patches, T, self.mix_dim)
            doy_mix = d

        # Temporal mixing in 768

        """
        if self.use_CBM:
            with torch.no_grad():
                mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]
        else:
            mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]
        """
        mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]

        last_idx  = (seq_len - 1).clamp_min(0)                      # [B]
        idx_bp    = last_idx.unsqueeze(1).expand(B, P).reshape(B*P) # [B*P]
        row_ids   = torch.arange(B*P, device=mix_out.device)
        last_768  = mix_out[row_ids, idx_bp, :]                     # [B*P,768]
        last_768  = self.vision.ln_post(last_768) if self.use_ln_norm_patch else last_768
        patch_vec = last_768 @ self.vision.proj                     # [B*P,512]

        # [B*P, 512] -> [B, P, 512]
        patch_feats = patch_vec.view(B, self.num_patches, self.embed_dim)

        # [B, P, 512] -> [B, 512, H_p, W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim) \
                                .permute(0, 3, 1, 2).contiguous()

        """
        if self.use_CBM:
            patch_flat = patch_feats.permute(0,2,3,1).contiguous().view(B * self.num_patches, self.embed_dim)
            with torch.no_grad():
                z_pre, z = self.sae.net.encode(patch_flat)   # concepts
            patch_feats = z.view(B, self.H_patch, self.W_patch, -1).permute(0,3,1,2).contiguous()  # [B, C_concept, H_p, W_p]
            
            if self.log_concepts:
                self.last_concept_map_raw = patch_feats.detach().clone()

            if self.editing_vector is not None:
                patch_feats *= self.editing_vector.view(1, -1, 1, 1)

            if self.log_concepts:
                self.last_concept_map = patch_feats.detach()"""


        out = self.head(patch_feats)
        out = rearrange(out, 'b (c r1 r2) h w -> b c (h r1) (w r2)', r1=int(self.upscale_factor**0.5), r2=int(self.upscale_factor**0.5))
        out = F.interpolate(
            out, size=(H, W) if not self.ds_labels else (self.out_H, self.out_W),
            mode="bilinear", align_corners=False
        )
        return out

    @torch.no_grad()
    def encode_patches(self, batch, use_temp=False, doy=None, seq_len=None):
        """
        Extract per-time, per-patch MS-CLIP embeddings *before* temporal aggregation.

        Args:
            batch: [B, T, C, H, W] tensor (same normalization as segmentation training)

        Returns:
            patch_feats: [B, T, P, D] where D=self.embed_dim (512),
                         P = H_patch * W_patch
        """
        assert batch.ndim == 5, f"Expected [B,T,C,H,W], got {batch.shape}"
        B, T, C, H, W = batch.shape

        x = batch.reshape(B * T, C, H, W)

        if self.vpt is not None:
            # Run VPT forward manually through the vision encoder
            B_T = x.shape[0]
            v = self.vision

            feat = v.conv1(x)                                              # [B*T, D, H_p, W_p]
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)         # [B*T, D, P]
            feat = feat.permute(0, 2, 1)                                   # [B*T, P, D]
            feat = torch.cat([
                v.class_embedding.unsqueeze(0).unsqueeze(0).expand(B_T, -1, -1),
                feat
            ], dim=1)                                                      # [B*T, 1+P, D]
            feat = feat + v.positional_embedding.unsqueeze(0)
            feat = v.ln_pre(feat)

            feat = self.vpt(feat)                                          # [1+P, B*T, D] LND
            pooled_feats = feat[:, 0]                                      # CLS [B*T, D]
            patch_feats  = feat[:, 1:]                                     # patches [B*T, P, D]
            pooled_feats    = v.ln_post(pooled_feats) if self.use_ln_norm_patch else pooled_feats
            pooled_feats = pooled_feats @ v.proj            # [B*T, 512]
        else:
            pooled_feats, patch_feats = self.msclip_model.image_encoder(x)

        if use_temp:

            if seq_len is None:
                # If caller doesn't provide true sequence lengths, assume all timesteps are valid.
                seq_len = torch.full((B,), T, device=batch.device, dtype=torch.long)

            t_idx = torch.arange(T, device=batch.device).unsqueeze(0)      # [1,T]
            valid_BT  = t_idx < seq_len.unsqueeze(1)                       # [B,T] True=valid
            P = self.num_patches
            valid_BPT = valid_BT.unsqueeze(1).expand(-1, P, -1)            # [B,P,T]
            valid_mask = valid_BPT.reshape(B * P, T)                       # [B*P,T]

            patch_feats = patch_feats.view(B, T, self.num_patches, self.mix_dim) \
                                .permute(0, 2, 1, 3).contiguous() \
                                .view(B * self.num_patches, T, self.mix_dim)          # [B*P, T, 768]

            # DOY for mixer (768-d)
            doy_mix = None
            if self.use_doy and (doy is not None):
                if doy.ndim > 2:
                    doy = doy.view(B, T, -1)[:, :, 0]
                assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
                d = self.doy_embed_mix(doy).unsqueeze(1).expand(-1, self.num_patches, -1, -1) \
                                        .reshape(B * self.num_patches, T, self.mix_dim)
                doy_mix = d

            mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]

            last_idx  = (seq_len - 1).clamp_min(0)                      # [B]
            idx_bp    = last_idx.unsqueeze(1).expand(B, P).reshape(B*P) # [B*P]
            row_ids   = torch.arange(B*P, device=mix_out.device)
            last_768  = mix_out[row_ids, idx_bp, :]                     # [B*P,768]
            last_768  = self.vision.ln_post(last_768) if self.use_ln_norm_patch else last_768
            patch_vec = last_768 @ self.vision.proj                     # [B*P,512]

            # [B*P, 512] -> [B, P, 512]
            patch_feats = patch_vec.view(B, self.num_patches, self.embed_dim)

            return patch_feats

        patch_feats = self.vision.ln_post(patch_feats) if self.use_ln_norm_patch else patch_feats                 # [B*T, P, 768]
        patch_feats = patch_feats @ self.vision.proj                    # [B*T, P, 512]

        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)
        return patch_feats