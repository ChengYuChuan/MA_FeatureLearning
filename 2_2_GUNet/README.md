# CubeNet

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Environment Variables Table](#Environment-Variables-Table)
- [Repository structure](#repository-structure)
- [License](#license)

# Setup
## Setting up the environment
### Conda:
LAP: `environment_LAP.yml`

Autoencoder: `environment.yml`
```sh
conda env create -f environment.yml
```
### venv:
```sh
python3.10 -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate.bat  # For Windows
```
Install Python dependencies:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Setting up the configuration file

A `.env` file is used for the configuration, and a template of it can be found in the `.env.nopath` file. Make a copy of this file and rename it `.env` with:

```sh
cp .env.nopath .env
```

Then fill out the fields with the values corresponding to your use case.

> :warning: **Note on the GROUP field**: It should be removed completely from the file if you intend to use the CNN model and not the G-CNN one.

# Usage

There are three different *use cases* possible of the model: **training without prior checkpoints**, **loading from checkpoints and not resuming training**, **loading from checkpoints and resuming training**. The use case can be chosen through the environment variables.

## 1. Training without prior checkpoints

The following variables should be set as:

```sh
LOAD_FROM_CHECKPOINTS=False
SHOULD_TRAIN=True
```

## 2. Loading from checkpoints and not resuming training

The following variables should be set as:


```sh
LOAD_FROM_CHECKPOINTS=True
CHECKPOINTS_PATH=/path/to/you/checkpoints
SHOULD_TRAIN=False
```

## 3. Loading from checkpoints and resuming training

The following variables should be set as:

```sh
LOAD_FROM_CHECKPOINTS=True
CHECKPOINTS_PATH=/path/to/you/checkpoints
SHOULD_TRAIN=True
```

## Using the model

After setting the variables to the desired use case, to run the model, use inside the activated environment:

Autoencoder:
```sh
python pretrain_encoder_main.py
```
LAP:
```sh
python CubeLAPwMLP_main.py
```

# Outputs

## Logs
- Execution logs can be found in the `.\logs` folder creted during installation.
- Tensorboard logs can be found in the `.\logs_tf` folder, inside subfolders named with the pattern `LOG_NAME-nb_layers-learning_rate-clip_value`, with `LOG_NAME` specified as a variable.

# Environment Variables Table

| Variable Name | Description                                                                                                                                                                  | Suggested Default |
| :-- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
| \#\# 1. Training \& Model Behavior |                                                                                                                                                                              |                   |
| `SHOULD_TRAIN` | Boolean to control whether the training process should be performed.                                                                                                         | `True`            |
| `LOAD_FROM_CHECKPOINTS` | Boolean to load model weights from a saved checkpoint.                                                                                                                       | `False`           |
| `CHECKPOINTS_PATH` | Path to the checkpoint file to load model weights from.                                                                                                                      | `None`            |
| \#\# 2. Dataset Settings |                                                                                                                                                                              |                   |
| `PATH_TO_DATA` | Path to the folder containing the dataset.                                                                                                                                   | `./data`          |
| `BATCH_SIZE` | Batch size for the dataloader.                                                                                                                                               | `16`              |
| `NUM_WORKERS` | Number of CPU workers for the dataloader.                                                                                                                                    | `4`               |
| `NUM_CELLS` | Number of cells in a worm. Only when you are doing LAP part you would need it. In our case, the max of num is `558`                                                          | `20`              |
| `SEED` | Random seed for train/validation splits to ensure reproducibility.                                                                                                           | `42`              |
| \#\# 3. Model \& Group Settings |                                                                                                                                                                              |                   |
| `GROUP` | Name of the group for Group Equivariant CNNs (G-CNNs).Usually it's `S4` or `T4` **Remove this field if using a standard CNN.**                                               | `None`            |
| `GROUP_DIM` | Dimension of the group for G-CNNs. `24` for `S4`, `12` for `T4`                                                                                                              | `None`            |
| `IN_CHANNELS` | Number of input channels for the model (e.g., 1 for grayscale images).                                                                                                       | `1`               |
| `OUT_CHANNELS` | Number of output channels for the model, typically equal to the number of classes. In Autoencoder, we need to set it as `1`. In LAP, it should be `None`                     | `None`            |
| `NONLIN` | Non-linearity activation function. Options: "relu", "leaky-relu", or "elu".                                                                                                  | `leaky-relu`      |
| `NORMALIZATION` | Type of normalization layer, e.g., "bn" (Batch Norm) or "in" (Instance Norm).                                                                                                | `bn`              |
| `DIVIDER` | An integer divisor to reduce the number of channels in each layer, decreasing the total model parameters. If our feature map start from 16 in the encoder, it should be `4`. | `4`               |
| `MODEL_DEPTH` | Depth of the U-Net model.                                                                                                                                                    | `4`               |
| `DROPOUT` | Dropout rate.                                                                                                                                                                | `0.1`             |
| \#\# 4. Logs \& Saving |                                                                                                                                                                              |                   |
| `LOGS_DIR` | Path to the directory where Tensorboard logs will be saved.                                                                                                                  | `./logs`          |
| `LOG_NAME` | Name prefix for this specific run in Tensorboard and results folders.                                                                                                        | `default_run`     |
| \#\# 5. Loss Function \& Optimizer |                                                                                                                                                                              |                   |
| `LEARNING_RATE` | The learning rate for the optimizer.                                                                                                                                         | `0.001`           |
| `LR_PATIENCE` | Patience for the learning rate scheduler (epochs of no improvement before reducing LR). Used for `ReduceLROnPlateau`.                                                        | `5`               |
| `LR_FACTOR` | Factor by which the learning rate will be reduced (e.g., `new_lr = lr * factor`).                                                                                            | `0.1`             |
| `LR_MIN` | The lower bound on the learning rate.                                                                                                                                        | `1e-6`            |
| `DISTANCE_TYPE` | The distance metric used for the loss function, e.g., "MSE" (L2 Loss) or "L1". **It defines how to compute features distance between two worms.**                            | `MSE`             |
| `LAMBDA` | It's a parameter for Continuous interpolation of a piecewise constant function from paper: Differentiation of Blackbox Combinatorial Solvers.                                | `15`              |
| \#\# 6. Trainer Settings |                                                                                                                                                                              |                   |
| `EARLY_STOPPING` | Boolean to enable or disable the Early Stopping callback.                                                                                                                    | `True`            |
| `EARLY_STOPPING_PATIENCE` | Patience for Early Stopping (epochs of no improvement before stopping training).                                                                                             | `10`              |
| `GPUS` | Number or identifier of the GPU(s) to use.                                                                                                                                   | `1`               |
| `PRECISION` | GPU precision to use. Options: `16` (or `16-mixed`), `32`, `64`.                                                                                                             | `32`              |
| `MAX_EPOCHS` | Maximum number of epochs to train for.                                                                                                                                       | `50`              |
| `VAL_CHECK_INTERVAL` | Frequency of validation checks within an epoch (1.0 means once per epoch).                                                                                                   | `1.0`             |
| `LOG_EVERY_N_STEPS` | How often to log metrics every N steps.                                                                                                                                      | `50`              |
| `PROGRESS_BAR_REFRESH_RATE` | Refresh rate for the progress bar.                                                                                                                                           | `20`              |
| \#\# 7. Data Normalization |                                                                                                                                                                              |                   |
| `INTENSITY_MEAN` | You can find the global cells mean in my `.env` file. Just in case if we need global normalization. **Must be computed from your data.**                                     | `None`            |
| `INTENSITY_STD` | You can find the global cells Standard deviation in my `.env`. Just in case if we need global normalization. **Must be computed from your data.**                                                                         | `None`            |


# Repository structure

```sh
.
├── .DS_Store
├── CubeLAP_main.py
├── CubeLAPwMLP_main.py
├── CubeLAPwNdLinear.py
├── ENV_files
│   ├── .env
│   ├── .env.nopath
│   ├── .env_1
│   ├── .env_2
│   ├── .env_3
│   ├── .env_4
│   ├── .env_5
│   ├── .env_6
│   ├── .env_7
│   ├── .env_8
│   ├── .env_LAP
│   ├── requirements.txt
│   └── environment.yml
├── GUNet.sh
├── README.md
├── pretrain_encoder_main.py
├── src_GUNet
│   ├── .DS_Store
│   ├── architectures
│   │   ├── FEncoderwNdLinear.py
│   │   ├── FeatureEncoder.py
│   │   ├── FeatureEncoderNoMLP.py
│   │   ├── decoder.py
│   │   ├── dilated_dense.py
│   │   ├── encoder.py
│   │   └── unet.py
│   ├── groups
│   │   ├── S4_group.py
│   │   ├── T4_group.py
│   │   └── V_group.py
│   ├── layers
│   │   ├── convs.py
│   │   └── gconvs.py
│   ├── training
│   │   ├── datamodule.py
│   │   ├── datamodule_LAP.py
│   │   ├── lightningLAPNet.py
│   │   ├── lightningLAPNetwMLP.py
│   │   ├── lightningUnet.py
│   │   └── loss.py
│   └── utils
│       ├── CheckPoint
│       │   └── LoadCheckPoint.py
│       ├── concatenation
│       │   ├── OperationAndCat.py
│       │   └── ReshapedCat.py
│       ├── dropout
│       │   └── GaussianDropout.py
│       ├── helpers
│       │   └── helpers.py
│       ├── interpolation
│       │   ├── Interpolate.py
│       │   └── ReshapedInterpolate.py
│       ├── logging
│       │   ├── logging.py
│       │   └── loggingConfig.yml
│       ├── normalization
│       │   ├── ReshapedBatchNorm.py
│       │   ├── ReshapedSwitchNorm.py
│       │   └── SwitchNorm3d.py
│       ├── plots
│       │   └── plot.py
│       └── pooling
│           └── GPool3d.py
└──

17 directories, 56 files
```

# License

This repository is covered by the MIT license, but some exceptions apply, and are listed below:
- The file in `./src_GUNet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py) by Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li, and is covered by the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/), as mentionned also at the top of the file.