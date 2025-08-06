import logging
import torch
import torch.nn as nn
from typing import List, Optional, Union

from c_unet.architectures.encoder import EncoderBlock
from c_unet.architectures.decoder import DecoderBlock

class Unet(nn.Module):
    """
    U-net architecture supporting DoubleConv/ResNetBlock, Max/Avg pooling, Interpolate/TransposeConv upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        divider: int = 1,
        # Pooling
        pool_size: int = 2,
        pool_stride: Union[int, List[int]] = 2,
        pool_padding: Union[str, int] = 0,
        pool_type: str = "max",  # 'max' or 'avg'
        # Upsampling
        upsample_mode: str = "nearest",  # 'nearest', 'trilinear', etc.
        # Convolutional arguments
        stride: Union[int, List[int]] = 1,
        padding: Union[str, int] = 1,
        kernel_size: int = 3,
        # Architecture arguments
        model_depth: int = 4,
        basic_module: nn.Module = None,  # DoubleConv or ResNetBlock or ResBlockPNI
        # conv_layer_order: str = 'gcr',
        num_groups: int = 4,
        final_activation: Optional[str] = ""
    ):
        super(Unet, self).__init__()

        self.logger = logging.getLogger(__name__)

        # Model constants
        self.root_feat_maps = 32 // divider
        self.num_feat_maps = 16 // divider

        # pick basic module
        # if basic_module is None:
        #     basic_module = nn.Sequential
        if basic_module not in [DoubleConv, ResNetBlock, ResBlockPNI]:
            basic_module = ResBlockPNI

        # Encoder
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
            # conv_layer_order=conv_layer_order,
            num_groups=num_groups
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for depth in range(model_depth - 1, -1, -1):
            # compute in/out channels
            if depth == model_depth - 1:
                dec_in_channels = 2 ** (depth + 1) * self.root_feat_maps
            else:
                dec_in_channels = 2 ** (depth + 2) * self.root_feat_maps
            dec_out_channels = 2 ** (depth + 1) * self.root_feat_maps

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=dec_in_channels,
                    out_channels=dec_out_channels,
                    kernel_size=kernel_size,
                    scale_factor=(2, 2, 2),
                    basic_module=basic_module,
                    # conv_layer_order=conv_layer_order,
                    num_groups=num_groups,
                    mode=upsample_mode,
                    padding=padding,
                    upsample=True
                )
            )

        # output
        self.final_conv = nn.Conv3d(
            self.root_feat_maps * 2, out_channels, kernel_size=1
        )

        # final activation
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = nn.Identity()

    def forward(self, x):
        # Encoder
        x, downsampling_features = self.encoder(x)
        # Decoder
        for i, decoder in enumerate(self.decoder_blocks):
            # skip connection: from back layers of feature
            skip = downsampling_features[-(i + 2)]
            x = decoder(x, skip)
        x = self.final_conv(x)
        x = self.final_activation(x)
        self.logger.debug(f"Final output shape: {x.shape}")
        return x
