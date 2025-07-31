import torch
import numpy as np
import pytorch_lightning as pl
import torchio as tio
from pathlib import Path
from torch import Generator
from torch.utils.data import random_split


class LoadNumpyData(tio.Transform):
    """
    Lazy load numpy arrays and perform cell-level preprocessing.

    This transform loads .npy files containing 5D arrays of shape (N, 1, 32, 32, 32)
    and applies standardization transforms to each cell individually.

    Args:
        num_cells (int): Maximum number of cells to load from each .npy file

    Input Subject fields:
        - npy_path (str): Path to the .npy file to load

    Output Subject fields:
        - main_img (tio.ScalarImage): First cell as ScalarImage for TorchIO compatibility
        - cubes (torch.Tensor): All processed cells, shape (num_cells, 1, H, W, D)
        - perms (torch.Tensor): Permutation indices, shape (num_cells,)
        - inv_perms (torch.Tensor): Inverse permutation indices, shape (num_cells,)
    """
    def __init__(self, num_cells: int):
        super().__init__(p=1.0)
        self.num_cells = num_cells
        self.cell_transform = tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(),
            tio.EnsureShapeMultiple(8, method="pad"),
        ])

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        npy_path_raw = subject['npy_path']

        # 確保 npy_path 是 Path 物件
        if isinstance(npy_path_raw, str):
            npy_path = Path(npy_path_raw)
        else:
            npy_path = npy_path_raw

        # 載入資料（使用字串路徑）
        arr = np.load(str(npy_path))[:self.num_cells]
        processed = []
        for n in range(arr.shape[0]):
            t = torch.tensor(arr[n], dtype=torch.float32)
            t = self.cell_transform(t)
            processed.append(t)
        arr = torch.stack(processed, dim=0)

        subject['main_img'] = tio.ScalarImage(tensor=arr[0])
        subject['cubes'] = arr

        # 生成確定性排列（現在可以安全使用 .name）
        seed = hash(npy_path.name) % (2 ** 16)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.num_cells)
        inv = np.argsort(perm)
        subject['perms'] = torch.tensor(perm)
        subject['inv_perms'] = torch.tensor(inv)

        return subject


class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preprocessing 3D cell data.

    This DataModule handles loading .npy files containing 5D arrays of 3D cell images,
    applies preprocessing transforms, and creates train/validation splits for training.

    Args:
        task (str): Name of the task/directory containing .npy files
        subset_name (str): Substring filter for file selection (currently unused)
        batch_size (int): Batch size for data loaders. Defaults to 16
        num_cells (int): Maximum number of cells to load per file. Defaults to 558
        num_workers (int): Number of workers for data loading. Defaults to 0
        train_val_ratio (float): Ratio of data used for training (rest for validation). Defaults to 0.8
        seed (int): Random seed for reproducible train/val splits. Defaults to 1

    Data Structure:
        Each batch contains the following fields:
        - batch["cubes"]: torch.Tensor, shape (B, num_cells, 1, H, W, D) - Preprocessed 3D cell images
        - batch["perms"]: torch.Tensor, shape (B, num_cells) - Permutation indices for each sample
        - batch["inv_perms"]: torch.Tensor, shape (B, num_cells) - Inverse permutation indices
        - batch["main_img"]: tio.ScalarImage - First cell as ScalarImage (TorchIO compatibility)
        - batch["npy_path"]: list[str] - Original file paths, length B
    """

    def __init__(self,
                 task: str,
                 subset_name: str = "",
                 batch_size: int = 16,
                 num_cells: int = 558,
                 num_workers: int = 0,
                 train_val_ratio: float = 0.8,
                 seed: int = 1):
        super().__init__()
        self.dataset_dir = Path(task)
        self.subset_name = subset_name
        self.batch_size = batch_size
        self.num_cells = num_cells
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.seed = seed
        self.full_dataset = None

    def prepare_data(self):
        """
        Performs one-time global data validation and preparation.

        This method is called only once per node and should not set instance attributes.
        It validates:
        - Data directory existence
        - Presence of .npy files
        - Correct data format (5D arrays)

        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If no .npy files found or incorrect data format
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.dataset_dir}")
        npy_files = list(self.dataset_dir.glob("*.npy"))
        if not npy_files:
            raise ValueError(f"No .npy files in {self.dataset_dir}")
        sample = np.load(npy_files[0])
        if sample.ndim != 5:
            raise ValueError(f"Expected 5D array, got {sample.ndim}D from {npy_files[0]}")

    def setup(self, stage=None):
        """
        Sets up datasets for training and validation.

        This method creates TorchIO Subjects for each .npy file and applies lazy loading
        transforms. The full dataset is then split into training and validation sets
        based on the train_val_ratio parameter.

        Args:
            stage (str, optional): Training stage ('fit', 'validate', 'test', 'predict')

        Creates:
            - self.full_dataset: Complete dataset with lazy loading transforms
            - self.train_set: Training subset after random split
            - self.val_set: Validation subset after random split

        Transform Pipeline:
            1. LoadNumpyData: Loads .npy files and applies cell-level preprocessing
               - ToCanonical: Ensures consistent orientation
               - ZNormalization: Zero-mean unit-variance normalization
               - EnsureShapeMultiple: Pads to ensure shape divisible by 8
            2. Generates deterministic permutations based on filename hash
        """
        if self.full_dataset is not None:
            return

        npy_files = sorted(self.dataset_dir.glob("*.npy"))
        subjects = []

        for npy_path in npy_files:
            # 建立符合 TorchIO 要求的 Subject
            dummy_tensor = torch.zeros(1, 32, 32, 32, dtype=torch.float32)

            subject = tio.Subject(
                placeholder_img=tio.ScalarImage(tensor=dummy_tensor),
                npy_path=str(npy_path)
            )
            subjects.append(subject)

        transform = tio.Compose([
            LoadNumpyData(num_cells=self.num_cells),
        ])

        self.full_dataset = tio.SubjectsDataset(subjects, transform=transform)

        # 資料分割
        total = len(self.full_dataset)
        val_len = int((1 - self.train_val_ratio) * total)
        train_len = total - val_len
        self.train_set, self.val_set = random_split(
            self.full_dataset,
            [train_len, val_len],
            generator=Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        """
        Creates training data loader.

        Returns:
            tio.SubjectsLoader: Training data loader with shuffling enabled

        Batch Structure:
            batch["cubes"].shape       # torch.Tensor, shape: (B, num_cells, 1, H, W, D)
            batch["perms"].shape       # torch.Tensor, shape: (B, num_cells)
            batch["inv_perms"].shape   # torch.Tensor, shape: (B, num_cells)
            batch["main_img"]          # tio.ScalarImage batch
            batch["npy_path"]          # list[str], length B
        """
        return tio.SubjectsLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        Creates validation data loader.

        Returns:
            tio.SubjectsLoader: Validation data loader without shuffling

        Batch Structure:
            batch["cubes"].shape       # torch.Tensor, shape: (B, num_cells, 1, H, W, D)
            batch["perms"].shape       # torch.Tensor, shape: (B, num_cells)
            batch["inv_perms"].shape   # torch.Tensor, shape: (B, num_cells)
            batch["main_img"]          # tio.ScalarImage batch
            batch["npy_path"]          # list[str], length B
        """
        return tio.SubjectsLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
