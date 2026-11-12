import torch
import torch.nn as nn
import math

from experiments.concept_bottleneck.models.clearclip import ClearCLIPLastBlock

class VPTAdapter(nn.Module):
    """
    Deep Visual Prompt Tuning adapter for OpenCLIP ViT (NLD format).
    Inserts learnable prompt tokens at the input of the first total_d_layer + 1 transformer blocks;
    later blocks keep the prompt tokens output by the previous block (total_d_layer=0 -> shallow VPT).
    With strip_before_clearclip=True, prompts are removed before the first ClearCLIP block, so that block
    only attends over CLS + patches (set total_d_layer below that block's index to avoid unused prompts).
    """
    def __init__(
        self,
        vision: nn.Module,
        num_tokens: int = 20,
        total_d_layer: int = 11,
        prompt_dim: int = 768,
        dropout: float = 0.1,
        strip_before_clearclip: bool = False,
    ):
        super().__init__()
        self.num_tokens    = num_tokens
        self.total_d_layer = total_d_layer
        self.strip_before_clearclip = strip_before_clearclip
        # Plain reference (not registered as a submodule) so the frozen ViT blocks are not duplicated
        # in state_dict / parameters; the live ModuleList also sees blocks patched later (e.g. ClearCLIP)
        object.__setattr__(self, "resblocks", vision.transformer.resblocks)
        self.num_layers    = len(self.resblocks)
        assert 0 <= total_d_layer < self.num_layers, \
            f"total_d_layer must be in [0, {self.num_layers - 1}], got {total_d_layer}"

        patch_size = vision.patch_size[0] if isinstance(vision.patch_size, tuple) else vision.patch_size
        val = math.sqrt(6. / float(3 * patch_size * patch_size + prompt_dim))

        # All prompts: [total_d_layer + 1, num_tokens, D]
        # index 0 = shallow, 1..total_d_layer = deep
        self.prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer + 1, num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.prompt_proj    = nn.Linear(prompt_dim, prompt_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        self.prompt_dropout = nn.Dropout(dropout)

    def _prompt_end(self) -> int:
        """Index of the first block that must not see prompts (num_layers if there is none)."""
        if self.strip_before_clearclip:
            for i, block in enumerate(self.resblocks):
                if isinstance(block, ClearCLIPLastBlock):
                    assert i > 0, "strip_before_clearclip needs at least one non-ClearCLIP block first"
                    return i
        return self.num_layers

    def _strip_prompts(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, :1], x[:, 1 + self.num_tokens:]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, L, D] NLD format, already after conv1 + pos_embed + ln_pre
           L = 1 + num_patches (CLS + patches)
        Returns: [N, L, D] with prompt tokens removed — same shape as input
        """
        N, L, D = x.shape  # N=batch, L=1+P, D=dim

        prompt_end = self._prompt_end()

        for i in range(self.num_layers):
            if i == prompt_end:
                x = self._strip_prompts(x)
            elif i < prompt_end and i <= self.total_d_layer:
                # Prepare prompt: [1, num_tokens, D] -> [N, num_tokens, D]
                prompt = self.prompt_dropout(
                    self.prompt_proj(self.prompt_embeddings[i])
                ).unsqueeze(0).expand(N, -1, -1)  # [N, num_tokens, D]

                if i == 0:
                    # Insert after CLS: [N, 1+num_tokens+P, D]
                    x = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
                else:
                    # Replace previous prompts: keep CLS and patch tokens
                    x = torch.cat([
                        x[:, :1],                        # CLS  [N, 1, D]
                        prompt,                           # new prompts [N, num_tokens, D]
                        x[:, 1 + self.num_tokens:],       # patches [N, P, D]
                    ], dim=1)
            # else: keep the prompt tokens output by the previous block

            x = self.resblocks[i](x)

        # Return [N, 1+P, D] — CLS + patches only (prompts may already be stripped)
        return x if prompt_end < self.num_layers else self._strip_prompts(x)
