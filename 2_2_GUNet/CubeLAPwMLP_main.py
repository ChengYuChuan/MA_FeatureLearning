import os
import sys
import ast
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
from datetime import datetime
from decouple import Config, RepositoryEnv

from src_GUNet.training.datamodule import DataModule
from src_GUNet.architectures.FeatureEncoder import FeatureEncoder
from src_GUNet.training.lightningLAPNetwMLP import LightningLAPNetwMLP
from src_GUNet.training.loss import build_loss
from src_GUNet.utils.logging.logging import configure_and_return_logger
from src_GUNet.utils.CheckPoint.LoadCheckPoint import LoadCheckPoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchinfo import summary

print(f"[DEBUG] CPU available to this job: {os.cpu_count()}")

if len(sys.argv) > 1:
    env_path = sys.argv[1]
else:
    env_path = '$YOUR_HOME_DIR/CubeLAP/.env'  # 預設值

config = Config(repository=RepositoryEnv(env_path))

def main(logger, args):
    logger.info(f"CONFIGURATION \n\n {args}")
    print("Running pretraining with configuration:")
    print(args)

    # DATA
    data = DataModule(
        args["PATH_TO_DATA"],
        batch_size=args["BATCH_SIZE"],
        num_workers=args["NUM_WORKERS"],
        num_cells=args["NUM_CELLS"],
        seed=args.get("SEED", 1),
        args=args
    )
    data.prepare_data()
    data.setup()

    # MODEL
    model = FeatureEncoder(
        args.get("GROUP"),
        args.get("GROUP_DIM"),
        args.get("IN_CHANNELS"),
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

    # Load Encoder Weight
    if args.get("LOAD_FROM_CHECKPOINTS"):
        LoadCheckPoint(model, args["CHECKPOINTS_PATH"])
    print("-" * 20 + "\n")

    # LOGGER
    log_name = f"FeatureMatching-{args.get('MODEL_DEPTH')}-{args.get('LEARNING_RATE')}"
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args["LOGS_DIR"],
        name=log_name,
        default_hp_metric=False
    )

    # LIGHTNING MODULE
    lightning_model = LightningLAPNetwMLP(
        criterion=build_loss(args),
        optimizer_class=torch.optim.AdamW,
        lapnet = model,
        learning_rate=args["LEARNING_RATE"],
        lr_patience=args["LR_PATIENCE"],
        lr_factor=args["LR_FACTOR"],
        lr_min=args["LR_MIN"],
        gradients_histograms=False
    )


    checkpoint_best = ModelCheckpoint(
        monitor="val/loss",
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

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=args["EARLY_STOPPING_PATIENCE"],
        verbose=True,
        mode="min"
    )
    # TRAINER
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args["GPUS"],
        precision=args["PRECISION"],
        max_epochs=args["MAX_EPOCHS"],
        val_check_interval=args["VAL_CHECK_INTERVAL"],
        log_every_n_steps=args["LOG_EVERY_N_STEPS"],
        enable_progress_bar=True,  # replace progress_bar_refresh_rate
        logger=tb_logger,
        benchmark=True,
        callbacks=[
                    TQDMProgressBar(refresh_rate=args["PROGRESS_BAR_REFRESH_RATE"]),
                    checkpoint_best,
                    checkpoint_last,
                    early_stop_callback
        ]
    )

    # TRAIN
    start = datetime.now()
    print("Training started at", start)
    trainer.fit(model=lightning_model, datamodule=data)
    print("Training finished in:", datetime.now() - start)
    logger.info("Training complete")


if __name__ == "__main__":
    logger = configure_and_return_logger('src_GUNet/utils/logging/loggingConfig.yml')
    args = {
        "SHOULD_TRAIN": True,
        "LOAD_FROM_CHECKPOINTS": config("LOAD_FROM_CHECKPOINTS", cast=bool, default=False),
        "CHECKPOINTS_PATH": config("CHECKPOINTS_PATH"),

        "PATH_TO_DATA": config("PATH_TO_DATA"),
        "BATCH_SIZE": config("BATCH_SIZE", cast=int),
        "NUM_WORKERS": config("NUM_WORKERS", cast=int),
        "NUM_CELLS" : config("NUM_CELLS",cast=int, default=558), # how many cells per worm would you like to compare?
        "SEED": config("SEED", default=1, cast=int),

        "GROUP": config("GROUP", default=None),
        "GROUP_DIM": config("GROUP_DIM", default=1, cast=int),
        "IN_CHANNELS": config("IN_CHANNELS", default=1, cast=int),
        "NONLIN": config("NONLIN", default="leaky-relu"),
        "NORMALIZATION": config("NORMALIZATION", default="bn"),
        "DIVIDER": config("DIVIDER", cast=int),
        "MODEL_DEPTH": config("MODEL_DEPTH", cast=int),
        "DROPOUT": config("DROPOUT", cast=float),

        "LOGS_DIR": config("LOGS_DIR"),
        "LOG_NAME": config("LOG_NAME"),

        "USE_MULTI_LAYER_MATCHING": config("USE_MULTI_LAYER_MATCHING", cast=bool, default=False),
        "LEARNING_RATE": config("LEARNING_RATE", cast=float),
        "LR_PATIENCE": config("LR_PATIENCE", cast=int),
        "LR_FACTOR": config("LR_FACTOR", cast=float),
        "LR_MIN": config("LR_MIN", cast=float),

        "EARLY_STOPPING_PATIENCE": config("EARLY_STOPPING_PATIENCE", cast=int, default=5),

        "GPUS": config("GPUS", cast=int),
        "PRECISION": config("PRECISION", default="32", cast=str),
        "MAX_EPOCHS": config("MAX_EPOCHS", cast=int),
        "VAL_CHECK_INTERVAL": config("VAL_CHECK_INTERVAL", cast=float),
        "LOG_EVERY_N_STEPS": config("LOG_EVERY_N_STEPS", cast=int),
        "PROGRESS_BAR_REFRESH_RATE": config("PROGRESS_BAR_REFRESH_RATE", cast=int),

        "DISTANCE_TYPE": config("DISTANCE_TYPE", default="MSE"),
        "LAMBDA": config("LAMBDA", default=20, cast=float),
    }

    main(logger, args)
