import logging
import torch.nn as nn
from typing import List, Optional, Union

from c_unet.utils.interpolation.Interpolate import Interpolate
from c_unet.utils.concatenation.OperationAndCat import OperationAndCat

from c_unet.layers.gconvs import GconvResBlock, FinalGroupConvolution
from c_unet.layers.convs import ConvBlock, ConvResBlock, FinalConvolution


class DecoderBlock(nn.Module):
    """Encoding path of a U-Net architecture

    Args:
        - out_channels (int): Number of output channels   

        - kernel_size (int): Size of the kernel. Defaults to 3.
        - stride (Union[int, List[int]]): Stride of the convolution. Defaults to 1.
        - padding (Union[str, int]): Zero-padding added to all three sides of the input. Defaults to 1.

        - tconv_kernel_size (int): Size of the kernel. Defaults to 3.
        - tconv_stride (Union[int, List[int]]): Stride of the upsampling. Defaults to 2.
        - pool_padding (Union[str, int]): Zero-padding added to all three sides of the input at upsampling. Defaults to 1.
        - output_padding (Union[str, int]): Additional size added to one side of each dimension in the output shape. Defaults to 1.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        - dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - model_depth (int): Depth of the encoding path. Defaults to 4.
        - num_feat_maps (int): Base multiplier for output channels numberfor multiplication. Defaults to 16.
        - final_activation (str): Name of the final activation to use. Defaults to sigmoid.

        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(
            self,
            # Channels
            out_channels: int,
            # Kernel arguments
            kernel_size: int = 3,
            padding: Union[str, int] = "same",
            stride: Union[int, List[int]] = 1,
            # Transpose convolution
            tconv_kernel_size: int = 3,
            tconv_stride: Union[int, List[int]] = 2,
            tconv_padding: Union[str, int] = 1,
            output_padding: Union[str, int] = 1,
            # Convolution arguments
            dropout: Optional[bool] = 0.1,
            bias: bool = True,
            dilation: int = 1,
            nonlinearity: Optional[str] = "relu",
            normalization: Optional[str] = "bn",
            # Model arguments
            model_depth: int = 4,
            num_feat_maps: int = 16,
            final_activation: Optional[str] = None,
            # Group arguments (by default, no group)
            group: Union[str, None] = None,
            group_dim: int = 0):
        super(DecoderBlock, self).__init__()

        self.num_feat_maps = num_feat_maps
        self.logger = logging.getLogger(__name__)

        self.module_dict = nn.ModuleDict()
        self.cat = OperationAndCat(logger=self.logger)
        self.group = group

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2**(depth + 1) * self.num_feat_maps

            self.module_dict[f"upsample_{depth}"] = Interpolate(
                dilation, tconv_kernel_size, tconv_stride, tconv_padding,
                output_padding)

            not_first_conv = False

            in_channels, inter_channels = feat_map_channels * 6, feat_map_channels * 2
            if group:
                self.module_dict[f"conv_block_{depth}"] = GconvResBlock(
                    group,
                    group_dim,
                    in_channels,
                    inter_channels,
                    inter_channels,
                    not_first_conv,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    dropout=dropout,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    normalization=normalization)
            else:
                self.module_dict[f"conv_block_{depth}"] = ConvResBlock(
                    in_channels,
                    inter_channels,
                    inter_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=bias,
                    dilation=dilation,
                    nonlinearity=nonlinearity,
                    normalization=normalization)

            if depth == 0:
                in_channels, inter_channels = feat_map_channels * 2, feat_map_channels * 2

                # The group dimension is reshaped into the channel one before the convolution in final_conv
                if group:
                    in_channels = in_channels * group_dim

                final_conv = ConvBlock(in_channels,
                                       inter_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       bias=bias,
                                       dilation=dilation,
                                       nonlinearity=nonlinearity,
                                       normalization=normalization)
                if group:
                    self.module_dict["final_conv"] = FinalGroupConvolution(
                        stable_convolution=final_conv,
                        inter_channels=inter_channels,
                        out_channels=out_channels,
                        final_activation=final_activation,
                        group=group,
                        group_dim=group_dim
                    )
                else:

                    self.module_dict["final_conv"] = FinalConvolution(
                        final_conv, inter_channels, out_channels,
                        final_activation)

    def forward(self, x, down_sampling_features):
        """
        Args:
            - x: input feature map
            - down_sampling_features: feature maps from encoder path
        Returns:
            - output feature map, the segmentation of the input image
        """
        # print("Decoder input shape:", x.shape) # debug
        for key, layer in self.module_dict.items():
            if key.startswith("upsample"):
                down_sampling_feature = down_sampling_features[int(key[-1])]
                # print(f"{key} - down feature shape: {down_sampling_feature.shape}") # debug
                x = self.cat(x, down_sampling_feature, key, layer, self.group)
            else:
                # print(f"{key} input shape: {x.shape}") # debug
                x = layer(x)
                # print(f"{key} output shape: {x.shape}") # debug
        return x