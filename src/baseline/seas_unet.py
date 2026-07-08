"""U-Net model for baseline wildfire forecasting."""
from typing import Optional

import segmentation_models_pytorch as smp
from torch import nn


class UNet(nn.Module):
    """U-Net module for wildfire forecasting."""
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        encoder="efficientnet-b5",
        out_H: Optional[int] = None,
        out_W: Optional[int] = None,
        **sat_kwargs,
    ):
        super().__init__()
        self.net = smp.UnetPlusPlus(encoder_name=encoder, in_channels=input_dim, classes=num_classes)
        self.out_H = out_H
        self.out_W = out_W

    def forward(self, x, *args):  # x here is B, T, H, W, C
        """Forward pass of the U-Net model."""

        # Encoder
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        assert T == 1, "Error Temporal Dimension"
        x = x.reshape(B * T, C, H, W)
        x = self.net(x)

        if self.out_H is not None and self.out_W is not None:
            x = nn.functional.interpolate(x, size=(self.out_H, self.out_W), mode="bilinear")
        return x
