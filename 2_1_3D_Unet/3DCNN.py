import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from datetime import datetime
from decouple import config

from src_3DUNet.training.datamodule import DataModule
from src_3DUNet.architectures.unet import Unet
from src_3DUNet.training.lightningUnet import LightningUnet
from src_3DUNet.utils.logging.logging import configure_and_return_logger


def main(logger, args):
    logger.info(f"CONFIGURATION \n\n {args}")
    print("Running pretraining with configuration:")
    print(args)

    # DATA
    data = DataModule(
        args["PATH_TO_DATA"],
        batch_size=args["BATCH_SIZE"],
        num_workers=args["NUM_WORKERS"],
        test_has_labels=False,
        seed=args.get("SEED", 1),
        args=args
    )
    data.prepare_data()

    # MODEL
    model = Unet(
        args.get("IN_CHANNELS"),
        args.get("OUT_CHANNELS"),
        final_activation=args.get("FINAL_ACTIVATION"),
        nonlinearity=args.get("NONLIN"),
        normalization=args.get("NORMALIZATION"),
        divider=args.get("DIVIDER"),
        model_depth=args.get("MODEL_DEPTH"),
        dropout=args.get("DROPOUT"),
    )

    print("="*80)
    print("Model Architecture:")
    print(model)
    print("="*80)

    # LOSS
    loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.MSELoss()

    # LOGGER
    log_name = f"AEPretrain-{args.get('MODEL_DEPTH')}-{args.get('LEARNING_RATE')}"
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args["LOGS_DIR"],
        name=log_name,
        default_hp_metric=False
    )

    # LIGHTNING MODULE
    lightning_model = LightningUnet(
        loss_fn,
        torch.optim.AdamW,
        model,
        learning_rate=args["LEARNING_RATE"],
        gradients_histograms=False
    )

    checkpoint_best = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{args['LOGS_DIR']}/{args['LOG_NAME']}/checkpoints",
        filename="best-epoch{epoch:02d}-val{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=f"{args['LOGS_DIR']}/{args['LOG_NAME']}/checkpoints",
        filename="last",
        save_last=True
    )

    # TRAINER
    trainer = pl.Trainer(
        gpus=args["GPUS"],
        precision=args["PRECISION"],
        max_epochs=args["MAX_EPOCHS"],
        val_check_interval=0.25,
        log_every_n_steps=100,
        progress_bar_refresh_rate=50,
        logger=tb_logger,
        benchmark=True,
        callbacks=[checkpoint_best, checkpoint_last]
    )

    # TRAIN
    start = datetime.now()
    print("Training started at", start)
    trainer.fit(model=lightning_model, datamodule=data)
    print("Training finished in:", datetime.now() - start)
    logger.info("Pretraining complete")


if __name__ == "__main__":
    logger = configure_and_return_logger('src_3DUNet/utils/logging/loggingConfig.yml')
    args = {
        "SHOULD_TRAIN": True,
        "PATH_TO_DATA": config("PATH_TO_DATA"),
        "SUBSET_NAME": config("SUBSET_NAME", default=None),
        "BATCH_SIZE": config("BATCH_SIZE", cast=int),
        "NUM_WORKERS": config("NUM_WORKERS", cast=int),
        "SEED": config("SEED", default=1, cast=int),

        "GROUP": config("GROUP", default=None),
        "GROUP_DIM": config("GROUP_DIM", default=1, cast=int),
        "IN_CHANNELS": config("IN_CHANNELS", default=1, cast=int),
        "OUT_CHANNELS": config("OUT_CHANNELS", default=1, cast=int),
        "FINAL_ACTIVATION": config("FINAL_ACTIVATION", default=None),
        "NONLIN": config("NONLIN", default="leaky-relu"),
        "NORMALIZATION": config("NORMALIZATION", default="bn"),
        "DIVIDER": config("DIVIDER", cast=int),
        "MODEL_DEPTH": config("MODEL_DEPTH", cast=int),
        "DROPOUT": config("DROPOUT", cast=float),

        "LOGS_DIR": config("LOGS_DIR"),
        "LOG_NAME": config("LOG_NAME"),

        "LEARNING_RATE": config("LEARNING_RATE", cast=float),
        "GPUS": config("GPUS", cast=int),
        "PRECISION": config("PRECISION", default=32, cast=int),
        "MAX_EPOCHS": config("MAX_EPOCHS", cast=int),
    }

    main(logger, args)