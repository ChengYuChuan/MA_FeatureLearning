import torchio as tio
import pytorch_lightning as pl
import numpy as np
import torch

from torch import Generator as Generator
from torch.utils.data import random_split, DataLoader
from pathlib import Path


class ApplyPermutation(tio.Transform):
    """Applies a deterministic permutation to the cube dimension of the given image tensor in a Subject."""
    def __init__(self, keys=("cubes",), p=1.0):
        super().__init__(p)
        self.keys = keys

    def apply_transform(self, subject):
        seed = int(subject["perm_seed"].item())
        rng = np.random.default_rng(seed)

        for key in self.keys:
            data = subject[key].data  # (N, 1, 32, 32, 32)
            num_cells = data.shape[0]
            perm = rng.permutation(num_cells)
            inv_perm = np.argsort(perm)

            subject[key].data = data[perm]
            subject["perms"] = torch.tensor(perm)
            subject["inv_perms"] = torch.tensor(inv_perm)

        return subject

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
        - test_has_labels (bool); indicates whether or not to look for and download labels for test dataset
    """
    def __init__(self,
                 task: str,
                 subset_name: str = "",
                 batch_size: int = 16,
                 num_cells: int = 558,
                 num_workers: int = 0,
                 train_val_ratio: float = 0.8,
                 seed: int = 1,
                 args: dict = None):
        super().__init__()
        self.task = task
        self.subset_name = subset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_cells = num_cells
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.seed = seed
        self.subjects = None
        self.preprocess = None
        self.transform = None

        self.full_dataset = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.args = args or {}

    def prepare_data(self):
        """
        Creates Subject instances with the image, label and laterality for training
        and validation subjects, and for test subjects
        """
        path = Path(self.dataset_dir)
        npy_files = sorted(path.glob("*.npy"))
        subjects = []
        # setup Transform
        cell_transform = tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(),
            tio.EnsureShapeMultiple(8, method="pad"),
            # something else transform
        ])

        for i, npy_path in enumerate(npy_files):
            arr = np.load(npy_path)[:self.num_cells]  # (N, 1, 32, 32, 32)
            processed = []
            for n in range(arr.shape[0]):
                t = torch.tensor(arr[n], dtype=torch.float32)  # (1, 32, 32, 32)
                t = cell_transform(t)
                processed.append(t)
            arr = torch.stack(processed, dim=0)  # (N, 1, 32, 32, 32)
            subject_seed = hash(npy_path.name) % (2 ** 16)
            subject = tio.Subject(
                main_img=tio.ScalarImage(tensor=arr[0]), # arr[0] shape: (1, 32, 32, 32) just get over the limitation. it's dummy one
                cubes=arr,
                file_name=npy_path.stem,
                perm_seed=torch.tensor(subject_seed)
            )
            subject = ApplyPermutation(keys=("cubes",))(subject)
            subjects.append(subject)

        self.full_dataset = tio.SubjectsDataset(subjects)

    def setup(self, stage=None):
        """
        Setups the data in three SubjectsDatasets (training, validation and test).
        The training data is split randomly between training and validation.


        batch["cubes"].shape       # torch.Tensor, shape: (B, num_cells, 1, 32, 32, 32)
        batch["perms"].shape       # torch.Tensor, shape: (B, num_cells)
        batch["inv_perms"].shape   # torch.Tensor, shape: (B, num_cells)
        batch["file_name"]         # list[str], 長度 B
        batch["perm_seed"]         # torch.Tensor, shape: (B,) or scalar per sample

        """
        # self.train_set = tio.SubjectsDataset(self.train_subjects)
        # self.val_set = tio.SubjectsDataset(self.val_subjects)
        # self.test_set = []

        # Already handled in prepare_data
        val_len = int((1 - self.train_val_ratio) * len(self.full_dataset))
        train_len = len(self.full_dataset) - val_len

        self.train_set, self.val_set = random_split(
            self.full_dataset,
            [train_len, val_len],
            generator=Generator().manual_seed(self.seed)
        )

        print(f"[INFO] Total subjects: {len(self.full_dataset)}")
        print(f"[INFO] Train set: {len(self.train_set)}")
        print(f"[INFO] Val set: {len(self.val_set)}")

    def train_dataloader(self):
        print(f"[DEBUG] Expected training steps per epoch: {len(self.train_set)}")
        return tio.SubjectsLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return tio.SubjectsLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
