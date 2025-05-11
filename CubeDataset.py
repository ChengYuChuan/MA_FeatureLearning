import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.stats import gennorm
from utils import get_logger
import sys

logger = get_logger('Dataset')
Cubesets = sys.argv[2] # "Cubes" or "MaskedCubes"
CubeSize = sys.argv[3] # "24" or "32"
STR_CubesSize = str(CubeSize)
CubeSize = int(CubeSize)

def get_train_loaders(transform=None, num_workers=0, batch_size=1, device='GPU'):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).
    Args:
        config:  a top level configuration object containing the 'loaders' key
    Returns:
        dict {
            'train': <train_loader>
            'val': <val_loader>
        }
    """

    logger.info('Creating training and validation set loaders...')

    folder_path = "/home/students/cheng/"
    folder_path = folder_path + Cubesets + STR_CubesSize

    # train_datasets = dataset_class.create_datasets(loaders_config, phase='train')
    train_dataset = CubeDataset(folder_path,transform=transform, split="train")
    # val_datasets = dataset_class.create_datasets(loaders_config, phase='val')
    val_dataset = CubeDataset(folder_path,transform=transform, split="val")
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')

    if torch.cuda.device_count() > 1 and not device == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}'
        )
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train':DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    }



def _create_mask(shape, sigma=2.1, beta=10):
    """
    Creates a 3D mask that only applies to the last three dimensions of the input shape.

    Args:
        shape (tuple): The shape of the input tensor. Can be (D, H, W), (1, D, H, W), or (1, 1, D, H, W).
        sigma (float): Scale parameter for the generalized normal distribution.
        beta (float): Shape parameter for the generalized normal distribution.

    Returns:
        np.ndarray: A mask of the same shape as the input, but only modifying the last three dimensions.
    """

    # Extract the last three dimensions (D, H, W)
    D, H, W = shape[-3:]

    # Create grid coordinates in the range [-2σ, 2σ]
    grid_coords_D = np.linspace(-2, 2, D)
    grid_coords_H = np.linspace(-2, 2, H)
    grid_coords_W = np.linspace(-2, 2, W)

    # Compute PDFs for each axis
    pdf_D = gennorm.pdf(grid_coords_D, beta, loc=0, scale=sigma)
    pdf_H = gennorm.pdf(grid_coords_H, beta, loc=0, scale=sigma)
    pdf_W = gennorm.pdf(grid_coords_W, beta, loc=0, scale=sigma)

    # Normalize PDFs to ensure the peak value is 1
    pdf_D /= pdf_D.max()
    pdf_H /= pdf_H.max()
    pdf_W /= pdf_W.max()

    # Create a 3D mask using broadcasting
    mask_3D = pdf_D[:, None, None] * pdf_H[None, :, None] * pdf_W[None, None, :]

    # Expand mask to match the input shape
    # The mask should have shape (D, H, W), but needs to be expanded for extra dimensions
    mask = np.ones(shape)  # Initialize with ones (no change for batch/channel dims)
    mask[-D:, -H:, -W:] = mask_3D  # Apply mask only to the last three dimensions

    return mask

class CubeDataset(Dataset):
    def __init__(self, folder_path, transform=None, split="train", train_ratio=0.8, val_ratio=0.1):
        self.transform = transform
        self.files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])

        total_files = len(self.files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        if split == "train":
            self.files = self.files[:train_end]
        elif split == "val":
            self.files = self.files[train_end:val_end]
        elif split == "test":
            self.files = self.files[val_end:]
        else:
            raise ValueError("Invalid split argument. Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 讀取 .npy 檔案並轉換為 float32
        sample = np.load(self.files[idx]).astype(np.float32)  # 原本 shape (24, 24, 24)

        # 這裡先將資料轉成 (1,1,24,24,24)
        # 若你希望由 ToTensor transform 負責擴展維度，可移除此行
        # sample = sample[np.newaxis, np.newaxis, ...]  # 結果 shape (1,1,24,24,24)

        # 檢查數據形狀
        if sample.shape != (CubeSize, CubeSize, CubeSize):
            print(f"Warning: Cube {os.path.basename(self.files[idx])} has shape {cube.shape}")

        # 生成與 sample 尺寸匹配的 mask
        # 應用 mask
        # mask = _create_mask(sample.shape)
        # sample = sample * mask

        if self.transform:
            sample = self.transform(sample)
            target_sample = self.transform(sample)

        else:
          sample = sample
          target_sample = sample

        return (sample, target_sample)