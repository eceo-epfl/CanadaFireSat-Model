import torch
import torch.nn as nn
import math

class VPTAdapter(nn.Module):
    """
    Deep Visual Prompt Tuning adapter for OpenCLIP ViT (NLD format).
    Inserts learnable prompt tokens at each transformer block.
    """
    def __init__(
        self,
        vision: nn.Module,
        num_tokens: int = 20,
        total_d_layer: int = 11,
        prompt_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision        = vision
        self.num_tokens    = num_tokens
        self.total_d_layer = total_d_layer
        self.resblocks     = vision.transformer.resblocks
        self.num_layers    = len(self.resblocks)

        patch_size = vision.patch_size[0] if isinstance(vision.patch_size, tuple) else vision.patch_size
        val = math.sqrt(6. / float(3 * patch_size * patch_size + prompt_dim))

        # All prompts: [num_layers, num_tokens, D]
        # index 0 = shallow, 1..total_d_layer = deep
        self.prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer + 1, num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.prompt_proj    = nn.Linear(prompt_dim, prompt_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        self.prompt_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, L, D] NLD format, already after conv1 + pos_embed + ln_pre
           L = 1 + num_patches (CLS + patches)
        Returns: [N, L, D] with prompt tokens removed — same shape as input
        """
        N, L, D = x.shape  # N=batch, L=1+P, D=dim

        for i in range(self.num_layers):
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

            x = self.resblocks[i](x)

        # Strip prompts: return [N, 1+P, D] — CLS + patches only
        return torch.cat([x[:, :1], x[:, 1 + self.num_tokens:]], dim=1)
