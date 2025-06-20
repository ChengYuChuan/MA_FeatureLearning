import logging
import torch
import torch.nn as nn

from typing import List, Optional, Union

from c_unet.architectures.encoder import EncoderBlock


class FeatureEncoder(nn.Module):
    """ FeatureEncoder architecture, that can be used either with normal convolutions, or with group convolutions.
    The available groups are defined in equiHippo/groups

    Args:
        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        - divider (int): Divides the base for the number
            of channels in the model. Must be a power of two between 1 and 16. Defulats to 1.

        - pool_size (int): Size of the pooling kernel. Defaults to 2.
        - pool_stride (Union[int, List[int]]): Stride of the pooling. Defaults to 2.
        - pool_padding (Union[str, int]): Zero-padding added to all three sides of the input at pooling. Defaults to 0.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - stride (Union[int, List[int]]): Stride of the convolution. Defaults to 1.
        - padding (Union[str, int]): Zero-padding added to all three sides of the input. Defaults to 1.
        - kernel_size (int): Size of the kernel. Defaults to 3.
        - bias (bool): If True, adds a learnable bias to the output. Defaults to True.
        - dilation (int): Spacing between kernel elements. Defaults to 1.

        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - model_depth (int): Depth of the encoding path. Defaults to 4.
        - final_activation (str): Name of the final activation to use. Defaults to sigmoid.

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """

    def __init__(
            self,
            # Group arguments
            group: Union[str, None],
            group_dim: int,
            # Channels arguments
            in_channels: int,
            divider: int = 1,
            # Pooling
            pool_size: int = 2,
            pool_stride: Union[int, List[int]] = 2,
            pool_padding: Union[str, int] = 0,
            pool_reduction: Optional[str] = "mean",
            pool_factor: Optional[int] = 2,
            # Convolutional arguments
            dropout: Optional[float] = 0.1,  # <-- dropout is a float
            stride: Union[int, List[int]] = 1,
            padding: Union[str, int] = "same",
            kernel_size: int = 3,
            bias: bool = True,
            dilation: int = 1,
            # Additional layers
            nonlinearity: Optional[str] = "relu",
            normalization: Optional[str] = "bn",
            # Architecture arguments
            model_depth=4):
        super(FeatureEncoder, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.group = group

        # Model constants
        self.root_feat_maps = 32 // divider
        self.num_feat_maps = 16 // divider

        self.encoder = EncoderBlock(in_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    pool_size=pool_size,
                                    pool_stride=pool_stride,
                                    pool_padding=pool_padding,
                                    pool_reduction=pool_reduction,
                                    pool_factor=pool_factor,
                                    dropout=dropout,
                                    bias=bias,
                                    dilation=dilation,
                                    nonlinearity=nonlinearity,
                                    normalization=normalization,
                                    model_depth=model_depth,
                                    root_feat_maps=self.root_feat_maps,
                                    group=group,
                                    group_dim=group_dim)

        # 步驟 2: 空間壓縮
        self.spatial_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # *** 變更重點：動態計算展平後的維度 ***
        # Encoder 輸出通道數 = 2**model_depth * (32 // divider)
        # 空間壓縮後維度 = 4*4*4 = 64
        encoder_out_channels = (2 ** model_depth) * self.root_feat_maps
        flattened_dim = encoder_out_channels * (4 ** 3)

        self.logger.info(f"Dynamically calculated flattened dimension for Linear layer: {flattened_dim}")

        # 步驟 3-5: 展平與全連接降維 (使用動態計算的維度)
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        """
        Args:
            - x: input feature map
        Returns:
            - output_vector: a 256-dimensional feature vector.
        """
        # 取得 encoder 的輸出特徵圖
        x, _ = self.encoder(x)  # 我們不再需要 downsampling_features

        # Step 1: Group Pooling
        # 如果是 G-CNN，則沿著群組維度進行最大池化，以實現不變性
        if self.group is not None:
            # G-CNN 的輸出張量維度通常是 (N, C, G, D, H, W)
            # 在群組維度 (dim=2) 上取最大值，將等變特徵轉為不變特徵
            # .max() 返回 (values, indices)，我們只需要 values
            x = torch.max(x, dim=2)[0]
            # 現在張量維度為 (N, C, D, H, W)，與標準 CNN 一致
            self.logger.debug(f"After Group Pooling, shape: {x.shape}")

        # Step2: Spatial Compression
        x = self.spatial_pool(x)
        self.logger.debug(f"After Spatial Pooling, shape: {x.shape}")

        # Step3: Flatten & FC Layers
        output_vector = self.projection_head(x)
        self.logger.debug(f"Final output vector shape: {output_vector.shape}")

        return output_vector
