import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional, Union

from src_3DUNet.architectures.buildingblocks import DoubleConv, ResNetBlock, InterpolateUpsampling, TransposeConvUpsampling

class Decoder(nn.Module):
    def __init__(
        self,
        model_depth: int,
        root_feat_maps: int,
        kernel_size: int = 3,
        basic_module: nn.Module = DoubleConv,
        conv_layer_order: str = 'gcr',
        num_groups: int = 4,
        mode: str = 'nearest',
        padding: int = 1,
        upsample: bool = True,
        upsample_mode: str = 'nearest'
    ):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        for depth in range(model_depth - 2, -1, -1):
            in_channels = 2 ** (depth + 2) * root_feat_maps
            out_channels = 2 ** (depth + 1) * root_feat_maps
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    scale_factor=(2, 2, 2),
                    basic_module=basic_module,
                    conv_layer_order=conv_layer_order,
                    num_groups=num_groups,
                    mode=upsample_mode,
                    padding=padding,
                    upsample=upsample
                )
            )

    def forward(self, x, encoder_features):
        # encoder_features: list, [enc1, enc2, ..., encN]
        for i, decoder in enumerate(self.decoder_blocks):
            skip = encoder_features[-(i + 2)]  # 倒序取對應 skip
            x = decoder(x, skip)
        return x


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block 支援 DoubleConv/ResNetBlock 及 Interpolate/TransposeConv 上採樣
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale_factor: Union[int, tuple] = (2, 2, 2),
        basic_module: nn.Module = DoubleConv,
        conv_layer_order: str = 'gcr',
        num_groups: int = 4,
        mode: str = 'nearest',
        padding: int = 1,
        upsample: bool = True,
    ):
        super().__init__()
        self.upsample = upsample

        # 決定 upsampling 方式
        if self.upsample:
            if basic_module == DoubleConv:
                self.upsampling = InterpolateUpsampling(mode=mode)
                self.joining = partial(self._joining, concat=True)
                up_out_channels = in_channels
            else:
                # ResNetBlock: 使用 ConvTranspose3d
                self.upsampling = TransposeConvUpsampling(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, scale_factor=scale_factor
                )
                self.joining = partial(self._joining, concat=False)
                up_out_channels = out_channels
        else:
            self.upsampling = nn.Identity()
            self.joining = partial(self._joining, concat=True)
            up_out_channels = in_channels

        # 建立 decoder block
        self.basic_module = basic_module(
            up_out_channels if basic_module == DoubleConv else out_channels,
            out_channels,
            encoder=False,
            kernel_size=kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding
        )

    def forward(self, x, encoder_feature):
        # 上採樣
        if isinstance(self.upsampling, InterpolateUpsampling):
            x = self.upsampling(x, output_size=encoder_feature.shape[2:])
        else:
            x = self.upsampling(x)

        # 合併 encoder skip connection
        x = self.joining(encoder_feature, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x
