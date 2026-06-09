from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _get_resblocks(image_encoder: nn.Module):
    """
    Return the ModuleList of ViT blocks in an OpenCLIP-style image encoder:
    image_encoder.model.visual.transformer.resblocks
    """
    return image_encoder.model.visual.transformer.resblocks

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """
    Implementation from torch webstie: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class ClearCLIPLastBlock(nn.Module):
    """
    Replacement for the LAST ViT block that implements ClearCLIP behavior:
      - uses the attention output only (X_attn)
      - optionally uses self-self attention (qq/kk/vv)
      - can drop residual and/or FFN
    """
    def __init__(
        self,
        src_block: nn.Module,
        attn_variant: str = "qq",         # "qk" (basic), "qq", "kk", "vv"
        keep_residual: bool = False,
        keep_ffn: bool = False,
    ):
        super().__init__()
        self.ln_1 = src_block.ln_1
        self.attn = src_block.attn                 # nn.MultiheadAttention
        self.ls_1 =src_block.ls_1
        self.ln_2 = src_block.ln_2
        self.mlp  = src_block.mlp
        self.ls_2 = src_block.ls_2

        self.attn_variant  = attn_variant.lower()
        self.keep_residual = keep_residual
        self.keep_ffn      = keep_ffn

        self.embed_dim  = self.attn.embed_dim
        self.num_heads  = self.attn.num_heads
        self.head_dim   = self.embed_dim // self.num_heads
        self.scale      = self.head_dim ** -0.5

    def _attention_only(self, x: torch.Tensor, attn_mask=None):
        N, L, C = x.shape

        def _reshape(z):
            return z.view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (N, heads, L, head_dim)

        q, k, v = self._proj_qkv(x)
        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                        dropout_p=self.attn.dropout if self.training else 0.0)

        out = out.permute(0, 2, 1, 3).contiguous().view(N, L, C)
        out = self.attn.out_proj(out)
        return out

    def _proj_qkv(self, x):
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.attn_variant == "qq":
            k = q
        elif self.attn_variant == "kk":
            q = k
        elif self.attn_variant == "vv":
            q, k = v, v
        return q, k, v

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        ClearCLIP at last block:
          z = LN1(x)
          a = Attn(z)      # with chosen self-self variant
          y = a            # discard residual
        """
        z = self.ln_1(x)
        a = self._attention_only(z, attn_mask=attn_mask)       # X_attn
        y = (x + self.ls_1(a)) if self.keep_residual else self.ls_1(a)
        if self.keep_ffn:
            y = y + self.ls_2(self.mlp(self.ln_2(y)))
        return y

def maybe_patch_clearclip(
    image_encoder: nn.Module,
    cfg: Dict[str, Any],
) -> int:
    """
    Patch the last N vision blocks to ClearCLIP behavior.
    cfg keys (all optional):
      - n_last: int = 1
      - attn_variant: str in {"qq","qk","kk","vv"}; default "qq"
      - keep_residual: bool = False
      - keep_ffn: bool = False
    Returns: number of blocks patched.
    """
    n_last       = int(cfg.get("n_last", 1))
    attn_variant = str(cfg.get("attn_variant", "qq")).lower()
    keep_res     = bool(cfg.get("keep_residual", False))
    keep_ffn     = bool(cfg.get("keep_ffn", False))

    resblocks = _get_resblocks(image_encoder)
    n = min(n_last, len(resblocks))
    for i in range(1, n + 1):
        print("patched ClearCLIP resblock")
        old = resblocks[-i]
        resblocks[-i] = ClearCLIPLastBlock(
            old,
            attn_variant=attn_variant,
            keep_residual=keep_res,
            keep_ffn=keep_ffn,
        )
    return n
