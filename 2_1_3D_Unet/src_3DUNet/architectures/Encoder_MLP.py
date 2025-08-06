import logging
import torch
import torch.nn as nn
from typing import List, Union

from src_3DUNet.architectures.encoder import EncoderBlock
from src_3DUNet.architectures.buildingblocks import ResBlockPNI, DoubleConv


class UnetEncoderMLP(nn.Module):
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
        latent_dim: int = 512
    ):
        super(UnetEncoderMLP, self).__init__()

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

        # compress the cubes
        self.spatial_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        flattened_dim = self.encoder_out_channels * (4 ** 3)

        # MLP
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 2048),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        # x: (B, N, C, D, H, W)
        batch_size, num_cubes, c, d, h, w = x.shape
        x = x.view(batch_size * num_cubes, c, d, h, w)

        # Encoder
        x, _ = self.encoder(x)

        # Compression + MLP
        x = self.spatial_pool(x)
        out = self.projection_head(x)
        out = out.view(batch_size, num_cubes, -1)
        return out