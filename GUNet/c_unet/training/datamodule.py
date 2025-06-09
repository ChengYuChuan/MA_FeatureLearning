import torchio as tio
import pytorch_lightning as pl
import numpy as np
import torch
import random

from torch import Generator as Generator
from torch.utils.data import random_split, DataLoader
from pathlib import Path

def _custom_yz_rotation(image):
    y_degree = 0 if random.random() < 0.5 else 180
    z_degree = 0 if random.random() < 0.5 else 180
    affine = tio.Affine(
        scales=1,
        degrees=(z_degree, y_degree, 0),  # degrees=(z, y, x)
        translation=(0, 0, 0),  # no translation
        default_pad_value=0,
        center='image'
    )
    return affine(image)

class DataModule(pl.LightningDataModule):
    """
    Data structure to use when traning with Pytorch Lightning

    Args:
        - task (str): name of the task to perform. Corresponds to the name of the folder where the data is stored.
        - subset_name (str): string preceding the .nii extension. 
            Only images with this substring will be used. Defaults to ""
        - batch_size (int): size of the batch. Defaults to 16
        - num_workers (int): number of workers for the dataloaders. Defaults to 0
        - train_val_ratio (float): ratio of the data to use in validation. Defaults to 0.7
    """
    def __init__(self,
                 task: str,
                 subset_name: str = "",
                 batch_size: int = 16,
                 num_workers: int = 0,
                 train_val_ratio: float = 0.8,
                 seed: int = 1,
                 args: dict = None):
        super().__init__()
        self.task = task
        self.subset_name = subset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.seed = seed
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.args = args or {}

    def get_max_shape(self, subjects):
        """
        Get the maximum shape in every direction over the given list of subjects

        Args:
            subjects: list of tio.Subject
        Returns:
            Tuple with the maximum shape in each dimension
        """
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.get_first_image().spatial_shape for s in dataset])
        return shapes.max(axis=0)
    
    def prepare_data(self):
        """
        Creates Subject instances with the image, label and laterality for training
        and validation subjects, and for test subjects
        """
        path = Path(self.dataset_dir)
        npy_files = sorted(path.glob("*.npy"))
        subjects = []

        for npy_path in npy_files:
            arr = np.load(npy_path)  # shape: (32, 32, 32)
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # shape: (1, 32, 32, 32)
            raw_image = tio.ScalarImage(tensor=tensor)

            rotated_image = tio.Compose([
                tio.RandomAffine(
                    scales=0,
                    degrees=(0, 0, 360),   # 只x軸隨機
                    default_pad_value=0,
                    p=1.0,
                ),
                tio.Lambda(_custom_yz_rotation)
            ])(raw_image)

            subject = tio.Subject(
                image=rotated_image,
                label=raw_image
            )
            subject = self.get_preprocessing_transform()(subject)
            subjects.append(subject)

        # ✅ 建立完整 dataset
        full_dataset = tio.SubjectsDataset(subjects)

        # ✅ 切割成 train / val
        val_len = int(0.2 * len(full_dataset))
        train_len = len(full_dataset) - val_len

        self.train_set, self.val_set = random_split(
            full_dataset,
            [train_len, val_len],
            generator=Generator().manual_seed(self.seed)
        )

        print(f"[INFO] Total subjects: {len(subjects)}")
        print(f"[INFO] Train set: {len(self.train_set)}")
        print(f"[INFO] Val set: {len(self.val_set)}")


    def get_preprocessing_transform(self):
        """
        Gets the composition of preprocessing transforms, which are applied on all subjects.
        They ensure a unique 'multiple of eight' shape for all of them, normalize the intensity
        values, and the laterality (left hippocampi are reversed to face right-ward), as well
        as the orientation. Labels are one-hot encoded.

        Returns:
            Composition of transformations
        """
        # 從 .env 讀取全域 normalization 參數
        # fixed_mean = self.args.get("INTENSITY_MEAN", 0.0)
        # fixed_std = self.args.get("INTENSITY_STD", 1.0)

        preprocess = tio.Compose([
            tio.ToCanonical(),  # 確保方向一致
            # tio.ZNormalization(mean=fixed_mean, std=fixed_std),  # 使用全域值 #TODO 如果要使用需要額外再寫一個函數
            tio.ZNormalization(),
            tio.EnsureShapeMultiple(8, method='pad')  # downsample 安全
        ])
        return preprocess

    def get_augmentation_transform(self):
        """
        Gets the composition of augmentation transforms, which are applied on some training subjects
        randomly.

        Returns:
            Composition of transformations
        """
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
        """
        Setups the data in three SubjectsDatasets (training, validation and test).
        The training data is split randomly between training and validation.
        """
        # self.train_set = tio.SubjectsDataset(self.train_subjects)
        # self.val_set = tio.SubjectsDataset(self.val_subjects)
        # self.test_set = []

        # Already handled in prepare_data
        pass

    def train_dataloader(self):
        print(f"[DEBUG] Expected training steps per epoch: {len(self.train_set)}")
        return DataLoader(self.train_set,
                          self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers)