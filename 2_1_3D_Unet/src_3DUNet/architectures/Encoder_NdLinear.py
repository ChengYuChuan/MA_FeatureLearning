import logging
import torch
import torch.nn as nn
from typing import List, Optional, Union

from c_unet.architectures.encoder import EncoderBlock
from src_3DUNet.architectures.buildingblocks import ResBlockPNI, DoubleConv
from ndlinear import NdLinear


class UnetEncoderNdLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        divider: int = 1,
        # Pooling
        pool_size: int = 2,
        pool_stride: Union[int, List[int]] = 2,
        pool_padding: Union[str, int] = 0,
        pool_type: str = "max",
        # Convolutional arguments
        stride: Union[int, List[int]] = 1,
        padding: Union[str, int] = 1,
        kernel_size: int = 3,
        # Architecture arguments
        model_depth: int = 4,
        basic_module: nn.Module = None,  # DoubleConv or ResNetBlock or ResBlockPNI
        num_groups: int = 4,
    ):
        super(UnetEncoderNdLinear, self).__init__()

        self.logger = logging.getLogger(__name__)

        # initial channels
        self.root_feat_maps = 32 // divider

        if basic_module is None:
            basic_module = ResBlockPNI

        # Encoder block
        self.encoder = EncoderBlock(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            pool_type=pool_type,
            model_depth=model_depth,
            root_feat_maps=self.root_feat_maps,
            basic_module=basic_module,
            num_groups=num_groups
        )

        # Encoder last channel
        self.encoder_out_channels = (2 ** (model_depth - 1)) * self.root_feat_maps

        # NdLinear
        self.NdLinear_head = nn.Sequential(
            NdLinear(input_dims=(64, 12, 8, 8, 8), hidden_size=(64, 12, 4, 4, 4)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            NdLinear(input_dims=(64, 12, 4, 4, 4), hidden_size=(32, 6, 4, 4, 4)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            NdLinear(input_dims=(32, 6, 4, 4, 4), hidden_size=(2, 6, 4, 4, 4)),
        )

    def forward(self, x):
        # x: (B, N, C, D, H, W)
        batch_size, num_cubes, c, d, h, w = x.shape
        x = x.view(batch_size * num_cubes, c, d, h, w)

        # Encoder
        x, _ = self.encoder(x)

        # Compression + NdLinear
        x = self.NdLinear_head(x)
        # flaten
        out = x.view(batch_size, num_cubes, -1)
        return out