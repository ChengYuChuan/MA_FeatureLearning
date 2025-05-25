import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.stats import gennorm
from utils import get_logger
import sys

logger = get_logger('Dataset')

def get_train_loaders(transform=None, num_workers=0, batch_size=1,folder_path = "/home/hd/hd_hd/hd_uu312/MergedCubes32", device='GPU'):
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

    train_dataset = CubeDataset(folder_path,transform=transform, split="train")
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

class CubeDataset(Dataset):
    def __init__(self, folder_path, transform=None, split="train", train_ratio=0.9, inference=False):
        self.transform = transform
        self.inference = inference
        self.files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
        total = len(self.files)
        split_idx = int(total * train_ratio)

        if split == "train":
            self.files = self.files[:split_idx]
        elif split == "val":
            self.files = self.files[split_idx:]
        elif split == "all":
            pass  # 使用全部 self.files，不需切割
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 讀取 .npy 檔案並轉換為 float32
        sample = np.load(self.files[idx]).astype(np.float32)  # 原本 shape (24, 24, 24)

        if self.transform:
            sample = self.transform(sample)
            target_sample = self.transform(sample)

        else:
          sample = sample
          target_sample = sample

        return (sample, target_sample)