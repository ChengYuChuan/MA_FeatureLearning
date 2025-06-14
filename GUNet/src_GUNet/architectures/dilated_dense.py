import logging
from typing import List, Optional, Union
import torch.nn as nn

from c_unet.layers.gconvs import GconvResBlock
from c_unet.layers.convs import ConvResBlock

class DilatedDenseBlock(nn.Module):
    """Dilated dense path for a U-Net architecture

    Args:
        - in_channels (int): Number of input channels
        - inter_channels (int): Number of intermediate channels
        - out_channels (int): Number of output channels

        - kernel_size (int): Size of the kernel. Defaults to 3.
        - stride (Union[int, List[int]]): Stride of the convolution. Defaults to 1.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - num_step_block (int): Number of blocks. Defaults to 3.
        - dilation_inscrease_step (int):Number of step per block with same dilation value. Defaults to 2.

        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(
            self,
            # Channels
            in_channels: int,
            inter_channels: int,
            out_channels: int,
            # Kernel arguments
            kernel_size: int = 3,
            stride: Union[int, List[int]] = 1,
            # Convolution arguments
            dropout: Optional[bool] = 0.1,
            bias: bool = True,
            nonlinearity: Optional[str] = "relu",
            normalization: Optional[str] = "bn",
            # Model
            num_steps_block: int = 3,
            dilation_increase_step: int = 2,
            # Group arguments (by default, no group)
            group: Union[str, None] = None,
            group_dim: int = 0):
        super(DilatedDenseBlock, self).__init__()

        self.logger = logging.getLogger(__name__)

        # <<< MODIFIED: use nn.ModuleList instead of plain list >>>
        self.module_list = nn.ModuleList()

        current_in_channels = in_channels  # <<< MODIFIED: track input channels through layers >>>

        for step_nb in range(num_steps_block):

            for _ in range(dilation_increase_step):
                dilation = 2**(step_nb)
                padding = "same"

                # <<< MODIFIED: use conv_block local var and append to ModuleList >>>
                if group:
                    conv_block = GconvResBlock(
                        group,
                        group_dim,
                        current_in_channels,
                        inter_channels,
                        out_channels,
                        is_first_conv=False,
                        kernel_size=kernel_size,
                        first_kernel_size=1,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        dropout=dropout,
                        first_padding=padding,
                        bias=bias,
                        nonlinearity=nonlinearity,
                        normalization=normalization
                    )
                else:
                    conv_block = ConvResBlock(
                        current_in_channels,
                        inter_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        first_kernel_size=1,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                        first_padding=padding,
                        dilation=dilation,
                        nonlinearity=nonlinearity,
                        normalization=normalization
                    )

                self.module_list.append(conv_block)
                current_in_channels = out_channels  # <<< MODIFIED: update in_channels >>>

    def forward(self, x):
        # <<< MODIFIED: explicitly apply each layer in module_list >>>
        for layer in self.module_list:
            x = layer(x)
        return x
