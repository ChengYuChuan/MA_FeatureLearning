import torchio as tio
import pytorch_lightning as pl
import numpy as np
import torch

from torch import Generator
from torch.utils.data import random_split, DataLoader
from pathlib import Path


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 task: str,
                 subset_name: str = "",
                 batch_size: int = 16,
                 num_workers: int = 0,
                 train_val_ratio: float = 0.8,
                 seed: int = 1,
                 args: dict = None,
                 augmentation_prob: float = 0.5):  # 新增參數控制 augmentation 機率
        super().__init__()
        self.task = task
        self.subset_name = subset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.seed = seed
        self.augmentation_prob = augmentation_prob
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.args = args or {}

    def prepare_data(self):
        """
        Creates Subject instances with the image, label and laterality for training
        and validation subjects, and for test subjects
        """
        pass

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.ToCanonical(),  # make sure that they are alinged
            tio.ZNormalization(),
            tio.EnsureShapeMultiple(8, method='pad')
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(p=0.1,
                             scales=0,
                             degrees=0,
                             translation=(0.05, 0.01, 0.05)),
            tio.RandomGamma(p=0.1, log_gamma=0.01),
            tio.RandomNoise(p=0.1, mean=0, std=(0, 0.01)),
        ])
        return augment

    def setup(self, stage=None):
        path = Path(self.dataset_dir)
        npy_files = sorted(path.glob("*.npy"))
        subjects = []

        for npy_path in npy_files:
            arr = np.load(npy_path)
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            raw_image = tio.ScalarImage(tensor=tensor)
            canonical_image = self.get_preprocessing_transform()(raw_image)

            subject = tio.Subject(
                image=canonical_image,
                label=canonical_image
            )
            subjects.append(subject)

        # build dataset
        full_dataset = tio.SubjectsDataset(subjects)

        # splitting train / val
        val_len = int((1 - self.train_val_ratio) * len(full_dataset))
        train_len = len(full_dataset) - val_len

        train_indices, val_indices = random_split(
            range(len(full_dataset)),
            [train_len, val_len],
            generator=Generator().manual_seed(self.seed)
        )

        train_subjects = [subjects[i] for i in train_indices]
        val_subjects = [subjects[i] for i in val_indices]

        # training use augmentation
        self.train_set = tio.SubjectsDataset(
            train_subjects,
            transform=self.get_augmentation_transform()
        )

        # validation won't use augmentation
        self.val_set = tio.SubjectsDataset(val_subjects)

        print(f"[INFO] Total subjects: {len(subjects)}")
        print(f"[INFO] Train set: {len(self.train_set)}")
        print(f"[INFO] Val set: {len(self.val_set)}")

        print(f"[INFO] Augmentation probability: {self.augmentation_prob}")

    def train_dataloader(self):
        print(f"[DEBUG] Expected training steps per epoch: {len(self.train_set)}")
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batch_size,
            drop_last=True,
            num_workers=self.num_workers
        )
