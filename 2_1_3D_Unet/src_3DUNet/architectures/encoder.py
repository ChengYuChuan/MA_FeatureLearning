import torch.nn as nn
from typing import List, Optional, Union

from src_3DUNet.architectures.buildingblocks import DoubleConv, ResNetBlock, ResBlockPNI

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: Union[int, List[int]] = 1,
        padding: Union[str, int] = 1,
        pool_size: int = 2,
        pool_stride: int = 2,
        pool_padding: int = 0,
        pool_type: str = 'max',
        model_depth: int = 4,
        root_feat_maps: int = 32,
        basic_module: nn.Module = DoubleConv,
        # conv_layer_order: str = 'gcr',
        num_groups: int = 4
    ):
        super(EncoderBlock, self).__init__()
        assert pool_type in ['max', 'avg'], "pool_type must be 'max' or 'avg'"
        self.root_feat_maps = root_feat_maps
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            self.module_dict[f"conv_block_{depth}"] = basic_module(
                in_channels,
                feat_map_channels,
                encoder=True,
                kernel_size=kernel_size,
                # order=conv_layer_order,
                num_groups=num_groups,
                padding=padding
            )
            in_channels = feat_map_channels
            if depth != model_depth - 1:
                if pool_type == 'max':
                    self.module_dict[f"pooling_{depth}"] = nn.MaxPool3d(
                        kernel_size=pool_size,
                        stride=pool_stride,
                        padding=pool_padding
                    )
                else:
                    self.module_dict[f"pooling_{depth}"] = nn.AvgPool3d(
                        kernel_size=pool_size,
                        stride=pool_stride,
                        padding=pool_padding
                    )

    def forward(self, x):
        down_sampling_features = []
        for key, layer in self.module_dict.items():
            if key.startswith("conv"):
                x = layer(x)
                down_sampling_features.append(x)
            elif key.startswith("pooling"):
                x = layer(x)
        return x, down_sampling_features
