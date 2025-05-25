import torch
import torch.nn as nn
import numpy as np
import sys
import math
# --- Dataset ---
from CubeDataset import CubeDataset, get_train_loaders
#---Transform---
import torchvision.transforms as transforms
import utils
# from torch_geometric.contrib.nn.models.rbcd_attack import LOSS_TYPE
# --- internal function ---
from utils import get_logger, load_checkpoint, create_optimizer, save_checkpoint, RunningAverage
from utils import _split_and_move_to_gpu, TensorboardFormatter
from transform import Standardize, RandomFlip, RandomRotate90, RandomRotate, ToTensor
from buildingblocks import SingleConv, DoubleConv, ResBlockPNI, ResNetBlock, Encoder, Decoder
from buildingblocks import create_encoders, create_decoders
from loss import get_loss_criterion
# --- Scheduler ---
import torch.optim.lr_scheduler as lr_scheduler
# --- AutoencoderTrainer ---
from AutoencoderTrainer import AutoencoderTrainer

from pathlib import Path


LossType = sys.argv[1] # "SSIMLoss" or "MSELoss"
DataFolder = sys.argv[2] # "Cubes" or "MaskedCubes"
PoolType = sys.argv[3] # 'avg' or 'max'
Learning_Rate = float(sys.argv[4]) # 0.0001

logger = get_logger('AutoencoderTrainer')

random_state = np.random.RandomState(66)  # 這樣才是正確的隨機狀態
transform_pipeline = transforms.Compose([
    Standardize(mean=0, std=1, min_max=True),
    RandomFlip(random_state),        # 預設隨機沿 (2,3,4) 翻轉
    RandomRotate90(random_state),      # 隨機以 90 度倍數旋轉
    RandomRotate(random_state, axes=[(2, 1)], angle_spectrum=45, mode='reflect'),
    ToTensor(expand_dims=True)        # 若資料為 (24,24,24) 則轉換成 (1,1,24,24,24)
])

base_dir = Path(__file__).resolve().parent.parent
print(f"Base Dir: {base_dir}")

folder_path = base_dir / DataFolder

loaders = get_train_loaders(transform=transform_pipeline,folder_path = folder_path ,num_workers=2, batch_size= 2) # training setting

train_dataset = CubeDataset(folder_path, transform=transform_pipeline, split="train")
val_dataset = CubeDataset(folder_path, transform=transform_pipeline, split="val")

# 打印各个 split 的数据集大小
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Define Autoencoder model, if you change the basic module, you should change layer order ALSO!!!!!
# DoubleConv gcr or ResNetBlock cge
# [32,64,128,256]
# [16,32,64,128]
#  ResBlockPNI gce,    ResNetBlock cge,    DoubleConv gcr
class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, f_maps=[16,32,64,128], layer_order='gce', num_groups=4, pool_type=PoolType):
        super(Autoencoder, self).__init__()
        # Create Encoders
        self.encoders = create_encoders(
            in_channels=in_channels,
            pool_type = pool_type,
            f_maps=f_maps,
            basic_module=ResBlockPNI,
            conv_kernel_size=3,
            conv_padding=1,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=2,
            downsample_mode='conv'
        )

        # Create Decoders
        self.decoders = create_decoders(
            f_maps=f_maps,
            basic_module=ResBlockPNI,
            conv_kernel_size=3,
            conv_padding=1,
            layer_order=layer_order,
            num_groups=num_groups,
            upsample=True
        )

        # Final 3D convolution layer to get the output
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1)

    def forward(self, x, return_logits=False, return_encoder_feats=False):
        encoder_features = []

        # Pass input through encoders
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)

        # Save intermediate feature maps from last two encoder layers
        last_3_feats = encoder_features[-3:]  # list of three feature maps

        # Pass input through decoders
        for i, decoder in enumerate(self.decoders):
            x = decoder(encoder_features[-(i + 2)], x)

        x = self.final_conv(x)

        if return_logits:
            if return_encoder_feats:
                return x, x, last_3_feats
            return x, x
        else:
            if return_encoder_feats:
                return x, last_3_feats
            return x


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


# Create an instance of your Autoencoder model
model = Autoencoder()

# Move the model to the appropriate device ('cuda:0' or 'cpu') before training
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

model.to(device)

num_params = count_parameters(model)
print(f"Total number of trainable parameters: {num_params}")

# --- Loss ---
# loss_criterion = SSIM3D(window_size=3, sigma=1.5, use_gaussian=True)
# loss_criterion = get_loss_criterion(name="MSELoss")
# loss_criterion = get_loss_criterion(name=LossType, window_size=window_size) # training setting
# loss_criterion = get_loss_criterion(name=LossType, window_size=window_size, return_msssim=True, alpha=alpha)
loss_criterion = get_loss_criterion(name=LossType)
# loss_criterion = get_loss_criterion(
#     name=LossType,
#     window_size=window_size,
#     levels=2,
#     weights=[0.5,0.5],
#     pool_type='max',      # 換成 'avg' 也可以
#     verbose=True          # 訓練階段你可以關掉
# )
loss_criterion.to(device)

# --- Evaluation ---
# eval_criterion = SSIM3D(window_size=3, sigma=1.5, use_gaussian=True)
# eval_criterion = get_loss_criterion(name=LossType, window_size=window_size) # training setting
# eval_criterion = get_loss_criterion(name=LossType, window_size=window_size, return_msssim=True, alpha=alpha)
eval_criterion = get_loss_criterion(name=LossType)
# eval_criterion = get_loss_criterion(
#     name=LossType,
#     window_size=window_size,
#     levels=2,
#     weights=[0.5, 0.5],
#     pool_type='max',      # 換成 'avg' 也可以
#     verbose=True          # 訓練階段你可以關掉
# )
eval_criterion.to(device)

# --- Optimizer ---
optimizer = create_optimizer('Adam', model, learning_rate=Learning_Rate, weight_decay=0.00001)
# optimizer = create_optimizer('AdamW', model, learning_rate=0.0001, weight_decay=0.00001)
# optimizer = create_optimizer('SGD', model, learning_rate=0.0001, weight_decay=0.00001)

# --- Scheduler ---
# lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5, min_lr=0.00001) # 每15個epoch衰減學習率
# lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                 mode='min',
#                                                 factor=0.5,
#                                                 patience=5,
#                                                 threshold=1e-4,
#                                                 verbose=True,
#                                                 cooldown=0,
#                                                 min_lr=1e-6
#                                                 )
# lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.000005)
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)


tensorboard_formatter = TensorboardFormatter()

total_data = 109368 #558*196= 109368
train_ratio = 0.8
batch_size = 2
num_epochs = 20

# Dynamic iterations
train_data = int(total_data * train_ratio)
iters_per_epoch = math.ceil(train_data / batch_size)
max_num_iterations = iters_per_epoch * num_epochs


trainer_config = {
  "checkpoint_dir" : "/home/students/cheng/3DUnet/CheckPoint_BS2_RBPNI_16_4Layers_CD_L1Loss_LR2e-4",
    "validate_after_iters": iters_per_epoch // 4,  # validate per 1/2 epoch → 22
    "log_after_iters": iters_per_epoch // 8,  # log per 1/4 epoch logging → 11
    "max_num_epochs": num_epochs,
    "max_num_iterations": max_num_iterations
}

trainer = AutoencoderTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                       eval_criterion=eval_criterion, loaders=loaders, tensorboard_formatter=tensorboard_formatter,
                       resume=None, pre_trained=None, **trainer_config)

trainer.fit()