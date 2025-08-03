import logging
import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.optim import Optimizer


class LightningLAPNetwMLP(pl.LightningModule):
    """
    Lightning Module for LAPNet training.

    Args:
        - criterion (nn.Module): Loss function used for both training and validation
        - optimizer_class (torch.optim.Optimizer): Optimizer class (e.g., torch.optim.Adam)
        - lapnet (nn.Module): Model (LAPNet or equivalent)
        - learning_rate (float): Initial learning rate
        - lr_patience (int): Scheduler's patience
        - lr_factor (float): Scheduler's factor
        - lr_min (float): Minimum learning rate
        - gradients_histograms (bool): Whether to log gradient histograms
    """

    def __init__(
            self,
            criterion: nn.Module,
            optimizer_class: Optimizer,
            lapnet: nn.Module,
            learning_rate: float = 0.001,
            lr_patience: int = 10,
            lr_factor: float = 0.5,
            lr_min: float = 1e-6,
            gradients_histograms: bool = False
    ):
        super().__init__()

        self.help_logger = logging.getLogger(__name__)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.LAPNet = lapnet

        self._learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.gradients_histograms = gradients_histograms

        self.save_hyperparameters(ignore=['criterion', 'optimizer_class', 'lapnet'])

    @property
    def learning_rate(self):
        if hasattr(self, 'hparams') and hasattr(self.hparams, 'learning_rate'):
            return self.hparams.learning_rate
        return getattr(self, '_learning_rate', 0.001)

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value
        if hasattr(self, 'hparams'):
            self.hparams.learning_rate = value

    def forward(self, x):
        return self.LAPNet(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_patience,
            factor=self.lr_factor,
            min_lr=self.lr_min,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        # ──────────────── batch ────────────────
        cubes = batch["cubes"]  # (2, N, 1, 32, 32, 32)
        inv_perms = batch["inv_perms"]  # (2, N)

        _, N, C, H, W, D = cubes.shape

        # ──────────────── add group dim ────────────────
        x = cubes.view(2 * N, C, H, W, D).to(self.device)  # (2N, 1, 32, 32, 32)
        x = x.unsqueeze(2)  # (2N, 1, 1, 32, 32, 32)

        # ──────────────── Encoder ────────────────
        final_feat = self.forward(x)  # (2N, 256)

        feats = final_feat.view(2, N, *final_feat.shape[1:])  # (2, N, 256)

        if torch.isnan(feats).any() or torch.isinf(feats).any():
            self.help_logger.warning(f"Invalid values detected in features at step {batch_idx}")
            return None

        loss, (row_ind, col_ind) = self.criterion(feats, inv_perm_A=inv_perms[0], inv_perm_B=inv_perms[1])

        if torch.isnan(loss) or torch.isinf(loss):
            self.help_logger.warning(f"Invalid loss detected at step {batch_idx}: {loss}")
            return None

        # ──────────────── Accuracy ────────────────
        acc = self._calculate_accuracy(row_ind, col_ind, inv_perms[0], inv_perms[1])

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        # ──────────────── batch ────────────────
        cubes = batch["cubes"]  # (2, N, 1, 32, 32, 32)
        inv_perms = batch["inv_perms"]  # (2, N)

        _, N, C, H, W, D = cubes.shape

        # ──────────────── get group dim ────────────────
        x = cubes.view(2 * N, C, H, W, D).to(self.device)  # (2N, 1, 32, 32, 32)
        x = x.unsqueeze(2)  # (2N, 1, 1, 32, 32, 32)

        # ──────────────── Encoder forward ────────────────
        final_feat = self.forward(x)

        feats = final_feat.view(2, N, *final_feat.shape[1:])

        # check the value stability
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            self.help_logger.warning(f"Invalid values detected in validation features at step {batch_idx}")
            return {"val_loss": torch.tensor(float('inf')), "val_acc": torch.tensor(0.0)}

        loss, (row_ind, col_ind) = self.criterion(feats, inv_perm_A=inv_perms[0], inv_perm_B=inv_perms[1])

        # ──────────────── Accuracy ────────────────
        acc = self._calculate_accuracy(row_ind, col_ind, inv_perms[0], inv_perms[1])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        return {"val_loss": loss, "val_acc": acc}

    def on_train_epoch_end(self):
        # record the current learning rate
        if self.optimizers():
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_epoch=True)

        if self.gradients_histograms and self.logger:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f"Grad of {name}", param.grad, self.current_epoch)
                    self.logger.experiment.add_scalar(
                        f"Mean of abs of grad of {name}",
                        torch.mean(torch.abs(param.grad)),
                        self.current_epoch
                    )

    def _calculate_accuracy(self, row_ind, col_ind, inv_perm_A, inv_perm_B):
        num_cells = len(row_ind)
        predicted_matching = np.zeros((num_cells, num_cells), dtype=np.float32)
        predicted_matching[row_ind, col_ind] = 1.0

        inv_perm_A = inv_perm_A.cpu().numpy()
        inv_perm_B = inv_perm_B.cpu().numpy()
        ideal_matching = np.zeros((num_cells, num_cells), dtype=np.float32)
        ideal_matching[inv_perm_A, inv_perm_B] = 1.0
        correct_matches = int((predicted_matching * ideal_matching).sum())
        accuracy = correct_matches / num_cells
        return accuracy
