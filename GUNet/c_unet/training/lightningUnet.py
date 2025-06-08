import logging
import pytorch_lightning as pl
import torch.nn as nn
import torchio as tio
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.optim import Optimizer


class LightningUnet(pl.LightningModule):
    """ Lightning model for automation of unet trainingg

    Args:
        - criterion (nn.functional): loss function
        - optimizer_class (torch.optim.Optimizer):
        - unet (nn.Module): model to use.
        - learning rate (float): learning rate. Defaults to 0.1
        - gradients_histograms: whether to log all parameters
            gradients as histograms and the mean of their absolute value as series. 
            Defaults to False.
    """
    def __init__(
            self,
            # Training arguments
            criterion: nn.Module,
            optimizer_class: Optimizer,
            unet: nn.Module,
            learning_rate: float = 0.001,
            gradients_histograms: bool = False,
            lr_patience: int = 10,
            lr_factor: float = 0.5,
            lr_min: float = 1e-6
            ):
        super(LightningUnet, self).__init__()

        self.help_logger = logging.getLogger(__name__)
        self.lr = learning_rate
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.unet = unet
        self.is_group = self.unet.group
        self.gradients_histograms = gradients_histograms

        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        if self.automatic_optimization or self._running_manual_backward:
            loss.backward(retain_graph=True, *args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor,
            min_lr=self.hparams.lr_min,
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

    def prepare_batch(self, batch):
        if self.is_group:
            image = batch['image'][tio.DATA].unsqueeze(1)
            return image, batch['label'][tio.DATA]
        else:
            return batch['image'][tio.DATA], batch['label'][tio.DATA]

    def infer_batch(self, batch):
        inputs, labels = self.prepare_batch(batch)
        outputs = self.forward(inputs)
        return outputs.float(), labels

    def training_step(self, batch, batch_idx):
        outputs, targets = self.infer_batch(batch)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, targets = self.infer_batch(batch)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.gradients_histograms:
            for name, params in self.named_parameters():
                if params.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"Grad of {name}", params.grad, self.current_epoch)
                    self.logger.experiment.add_scalar(
                        f"Mean of abs of grad of {name}",
                        torch.mean(torch.abs(params.grad)), self.current_epoch)
