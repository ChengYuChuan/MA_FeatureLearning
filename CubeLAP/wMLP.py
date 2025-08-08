import os
import sys
import ast
import torch
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
from datetime import datetime
from decouple import Config, RepositoryEnv
import matplotlib.pyplot as plt

from c_unet.training.datamodule import DataModule
from c_unet.architectures.FeatureEncoder import FeatureEncoder
from c_unet.training.lightningLAPNetwMLP import LightningLAPNetwMLP
from c_unet.training.loss import build_loss
from c_unet.utils.CheckPoint.LoadCheckPoint import LoadCheckPoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchinfo import summary

print(f"[DEBUG] CPU available to this job: {os.cpu_count()}")

if len(sys.argv) > 1:
    env_path = sys.argv[1]
else:
    env_path = '/home/students/cheng/CubeLAP/.env'  # 預設值

config = Config(repository=RepositoryEnv(env_path))


def main(logger, args):
    logger.info(f"CONFIGURATION \n\n {args}")
    print("Running pretraining with configuration:")
    print(args)

    # DATA
    data = DataModule(
        task=args["PATH_TO_DATA"],
        batch_size=args["BATCH_SIZE"],
        num_cells=args["NUM_CELLS"],
        num_workers=args["NUM_WORKERS"],
        train_val_ratio=args.get("TRAIN_VAL_RATIO", 0.8),
        seed=args.get("SEED", 1)
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

    print("=" * 80)
    print("Model Architecture:")
    print(model)
    print("=" * 80)

    # Load Encoder Weight
    if args.get("LOAD_FROM_CHECKPOINTS"):
        LoadCheckPoint(model, args["CHECKPOINTS_PATH"])
    print("-" * 20 + "\n")

    # LOGGER
    log_name = f"FeatureMatching-{args.get('LOG_NAME')}-{args.get('MODEL_DEPTH')}-{args.get('LEARNING_RATE')}"
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args["LOGS_DIR"],
        name=log_name,
        default_hp_metric=False
    )

    # LIGHTNING MODULE
    lightning_model = LightningLAPNetwMLP(
        criterion=build_loss(args),
        optimizer_class=torch.optim.AdamW,
        lapnet=model,
        learning_rate=args["LEARNING_RATE"],
        lr_patience=args["LR_PATIENCE"],
        lr_factor=args["LR_FACTOR"],
        lr_min=args["LR_MIN"],
        gradients_histograms=False
    )

    # 設定回調函數
    checkpoint_best = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{args['LOGS_DIR']}/{args['LOG_NAME']}/checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=f"{args['LOGS_DIR']}/{args['LOG_NAME']}/checkpoints",
        filename="last",
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args["EARLY_STOPPING_PATIENCE"],
        verbose=True,
        mode="min"
    )

    # TRAINER
    trainer = pl.Trainer(
        accelerator="gpu",
        gradient_clip_val=1.0,
        devices=args["GPUS"],
        precision=args["PRECISION"],
        max_epochs=args["MAX_EPOCHS"],
        val_check_interval=args["VAL_CHECK_INTERVAL"],
        log_every_n_steps=args["LOG_EVERY_N_STEPS"],
        enable_progress_bar=True,
        logger=tb_logger,
        benchmark=True,
        callbacks=[
            TQDMProgressBar(refresh_rate=args["PROGRESS_BAR_REFRESH_RATE"]),
            checkpoint_best,
            checkpoint_last,
            early_stop_callback
        ]
    )

    # ===== 完整的學習率尋找實現 =====
    if args.get("USE_LR_FINDER", True):
        print("=" * 50)
        print("Starting Learning Rate Finder...")
        print("=" * 50)

        try:
            # 嘗試不同的學習率尋找方法
            lr_finder = None
            suggested_lr = None

            # 方法 1: 使用 trainer.tune (PyTorch Lightning 2.0+)
            try:
                if hasattr(trainer, 'tune'):
                    result = trainer.tune(lightning_model, datamodule=data)
                    if hasattr(result, 'lr_find'):
                        lr_finder = result.lr_find
                        suggested_lr = lr_finder.suggestion() if hasattr(lr_finder, 'suggestion') else None
                        print("✓ Method 1 (trainer.tune) succeeded")
            except Exception as e1:
                print(f"Method 1 failed: {e1}")

            # 方法 2: 使用 trainer.tuner (較舊版本)
            if lr_finder is None:
                try:
                    if hasattr(trainer, 'tuner'):
                        lr_finder = trainer.tuner.lr_find(
                            lightning_model,
                            datamodule=data,
                            min_lr=args.get("LR_FINDER_MIN", 1e-8),
                            max_lr=args.get("LR_FINDER_MAX", 1e-2),
                            num_training=args.get("LR_FINDER_STEPS", 100)
                        )
                        suggested_lr = lr_finder.suggestion()
                        print("✓ Method 2 (trainer.tuner) succeeded")
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")

            # 方法 3: 手動創建 Tuner
            if lr_finder is None:
                try:
                    from pytorch_lightning.tuner import Tuner
                    tuner = Tuner(trainer)
                    lr_finder = tuner.lr_find(
                        lightning_model,
                        datamodule=data,
                        min_lr=args.get("LR_FINDER_MIN", 1e-8),
                        max_lr=args.get("LR_FINDER_MAX", 1e-2),
                        num_training=args.get("LR_FINDER_STEPS", 100)
                    )
                    suggested_lr = lr_finder.suggestion()
                    print("✓ Method 3 (manual Tuner) succeeded")
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")

            # 處理結果和保存圖表
            if suggested_lr is not None and lr_finder is not None:
                original_lr = args["LEARNING_RATE"]
                print(f"Original learning rate: {original_lr}")
                print(f"Suggested learning rate: {suggested_lr}")

                # 保存學習率曲線圖
                try:
                    fig = lr_finder.plot(suggest=True, show=False)
                    lr_plot_path = f"{args['LOGS_DIR']}/{args['LOG_NAME']}/lr_finder_plot.png"
                    os.makedirs(os.path.dirname(lr_plot_path), exist_ok=True)
                    fig.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Learning rate plot saved to: {lr_plot_path}")
                except Exception as plot_error:
                    print(f"Warning: Could not save LR plot: {plot_error}")

                # 智能學習率更新策略
                min_acceptable = args.get("LR_FINDER_MIN", 1e-8) * 10  # 比最小值大10倍
                max_acceptable = args.get("LR_FINDER_MAX", 1e-2) / 10  # 比最大值小10倍

                if min_acceptable <= suggested_lr <= max_acceptable:
                    # 使用 property setter 更新學習率
                    lightning_model.learning_rate = suggested_lr
                    print(f"✓ Updated to suggested learning rate: {suggested_lr}")
                elif suggested_lr < min_acceptable:
                    conservative_lr = min_acceptable
                    lightning_model.learning_rate = conservative_lr
                    print(f"⚠ Suggested LR too low, using conservative: {conservative_lr}")
                elif suggested_lr > max_acceptable:
                    conservative_lr = max_acceptable
                    lightning_model.learning_rate = conservative_lr
                    print(f"⚠ Suggested LR too high, using conservative: {conservative_lr}")
                else:
                    print(f"⚠ Using original learning rate: {original_lr}")
            else:
                print("⚠ Could not find optimal learning rate, using original")

        except Exception as e:
            print(f"⚠ All learning rate finder methods failed: {e}")
            print(f"Continuing with original learning rate: {args['LEARNING_RATE']}")

        print("=" * 50)

    # TRAIN
    start = datetime.now()
    print("Training started at", start)

    # 顯示最終學習率
    try:
        current_lr = lightning_model.learning_rate
        print(f"Final learning rate: {current_lr}")
    except AttributeError:
        print(f"Final learning rate: {args['LEARNING_RATE']}")

    # 顯示 PyTorch Lightning 版本信息
    print(f"PyTorch Lightning version: {pl.__version__}")

    try:
        trainer.fit(model=lightning_model, datamodule=data)
        print("Training finished in:", datetime.now() - start)
        logger.info("Training complete")

        # 訓練完成後的統計信息
        if hasattr(trainer, 'callback_metrics'):
            print("Final metrics:", dict(trainer.callback_metrics))

    except Exception as training_error:
        print(f"Training failed with error: {training_error}")
        logger.error(f"Training failed: {training_error}")

        # 增強的錯誤診斷
        error_str = str(training_error).lower()
        if "invalid numeric entries" in error_str or "nan" in error_str:
            print("\n" + "=" * 60)
            print("NUMERICAL STABILITY ERROR DETECTED!")
            print("Suggestions:")
            print("1. Try reducing learning rate further")
            print("2. Use full precision (PRECISION='32')")
            print("3. Increase batch size")
            print("4. Add gradient clipping (already enabled)")
            print("5. Check input data for NaN/Inf values")
            print("6. Consider using different loss function")
            print("=" * 60)
        elif "out of memory" in error_str or "cuda" in error_str:
            print("\n" + "=" * 60)
            print("GPU MEMORY ERROR DETECTED!")
            print("Suggestions:")
            print("1. Reduce batch size")
            print("2. Use gradient accumulation")
            print("3. Enable gradient checkpointing")
            print("4. Use mixed precision training")
            print("=" * 60)
        elif "dataloader" in error_str or "dataset" in error_str:
            print("\n" + "=" * 60)
            print("DATA LOADING ERROR DETECTED!")
            print("Suggestions:")
            print("1. Check data path and file permissions")
            print("2. Reduce num_workers")
            print("3. Verify data format and structure")
            print("=" * 60)

        raise training_error


if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    app_logger = logging.getLogger("MyTrainingApp")

    # 顯示系統信息
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

    args = {
        "SHOULD_TRAIN": True,
        "LOAD_FROM_CHECKPOINTS": config("LOAD_FROM_CHECKPOINTS", cast=bool, default=False),
        "CHECKPOINTS_PATH": config("CHECKPOINTS_PATH"),

        "PATH_TO_DATA": config("PATH_TO_DATA"),
        "BATCH_SIZE": config("BATCH_SIZE", cast=int),
        "NUM_WORKERS": config("NUM_WORKERS", cast=int),
        "NUM_CELLS": config("NUM_CELLS", cast=int, default=558),
        "TRAIN_VAL_RATIO": config("TRAIN_VAL_RATIO", cast=float, default=0.8),
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

        # 學習率尋找相關參數
        "USE_LR_FINDER": config("USE_LR_FINDER", cast=bool, default=True),
        "LR_FINDER_MIN": config("LR_FINDER_MIN", cast=float, default=1e-8),
        "LR_FINDER_MAX": config("LR_FINDER_MAX", cast=float, default=1e-2),
        "LR_FINDER_STEPS": config("LR_FINDER_STEPS", cast=int, default=100),

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

    main(app_logger, args)
