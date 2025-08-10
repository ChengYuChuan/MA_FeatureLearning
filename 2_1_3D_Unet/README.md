# 3D CNN

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

---

## 1. Training & Model Behavior

| Variable Name           | Description                                                          | Suggested Default |
| ----------------------- | -------------------------------------------------------------------- | ----------------- |
| `SHOULD_TRAIN`          | Boolean to control whether the training process should be performed. | `True`            |
| `LOAD_FROM_CHECKPOINTS` | Boolean to load model weights from a saved checkpoint.               | `False`           |
| `CHECKPOINTS_PATH`      | Path to the checkpoint file to load model weights from.              | `None`            |

---

## 2. Dataset Settings

| Variable Name     | Description                                                                           | Suggested Default |
| ----------------- | ------------------------------------------------------------------------------------- | ----------------- |
| `PATH_TO_DATA`    | Path to the folder containing the dataset.                                            | `/Cubes32`        |
| `CLASSES_NAME`    | Name of the label class in dataset.                                                   | `voxel`           |
| `TEST_HAS_LABELS` | Whether the test data contains labels. Useful for evaluating or inference-only setup. | `False`           |
| `BATCH_SIZE`      | Batch size for the dataloader.                                                        | `2`               |
| `NUM_WORKERS`     | Number of CPU workers for the dataloader.                                             | `2`               |

---

## 3. Model & Group Settings

| Variable Name   | Description                                                                                                   | Suggested Default |
| --------------- | ------------------------------------------------------------------------------------------------------------- | ----------------- |
| `IN_CHANNELS`   | Number of input channels for the model (e.g., 1 for grayscale images).                                        | `1`               |
| `OUT_CHANNELS`  | Output channels of model. For autoencoders it's usually `1`, but could vary (e.g., for MLP or NdLinear head). | `1`               |
| `NONLIN`        | Non-linearity activation function. Options: `relu`, `leaky-relu`, or `elu`.                                   | `leaky-relu`      |
| `NORMALIZATION` | Type of normalization layer, e.g., `bn` (Batch Norm) or `in` (Instance Norm).                                 | `bn`              |
| `MODEL_DEPTH`   | Depth of the U-Net model.                                                                                     | `3`               |
| `DIVIDER`       | Channel divisor used to scale down encoder/decoder width.                                                     | `4`               |
| `DROPOUT`       | Dropout rate.                                                                                                 | `0.0`             |

---

## 4. Logs & Saving

| Variable Name | Description                                                   | Suggested Default |
| ------------- | ------------------------------------------------------------- | ----------------- |
| `LOGS_DIR`    | Path to the directory where TensorBoard logs will be saved.   | `/TensorBoard`    |
| `LOG_NAME`    | Name prefix for this specific run in TensorBoard and results. | `Test`            |

---

## 5. Loss Function & Optimizer

| Variable Name   | Description                          | Suggested Default |
| --------------- | ------------------------------------ | ----------------- |
| `LEARNING_RATE` | The learning rate for the optimizer. | `0.0001`          |

---

## 6. Trainer Settings

| Variable Name | Description                                      | Suggested Default |
| ------------- | ------------------------------------------------ | ----------------- |
| `MAX_EPOCHS`  | Maximum number of epochs to train for.           | `10`              |
| `GPUS`        | Number or identifier of the GPU(s) to use.       | `1`               |
| `PRECISION`   | GPU precision to use. Options: `16`, `32`, `64`. | `16`              |

---

## 7. Data Normalization

| Variable Name    | Description                                                               | Suggested Default |
| ---------------- | ------------------------------------------------------------------------- | ----------------- |
| `INTENSITY_MEAN` | Global mean intensity value of all training data, used for normalization. | `38.90965`        |
| `INTENSITY_STD`  | Global standard deviation of intensity, used for normalization.           | `45.17005`        |

---

# Repository structure

```sh
.
├── .env
├── 3DCNN.py
├── CubeLAPwMLP_main.py
├── CubeLAPwNdLinear_main.py
├── README.md
├── environment.yml
├── requirements.txt
├── src_3DUNet
│   ├── architectures
│   │   ├── Encoder_MLP.py
│   │   ├── Encoder_NdLinear.py
│   │   ├── buildingblocks.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── unet.py
│   ├── layers
│   │   ├── convs.py
│   │   └── gconvs.py
│   ├── training
│   │   ├── datamodule.py
│   │   ├── datamodule_LAP.py
│   │   ├── lightningLAPNet.py
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

15 directories, 35 files
```

# License

This repository is covered by the MIT license, but some exceptions apply, and are listed below:
- The file in `./src_GUNet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py) by Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li, and is covered by the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/), as mentionned also at the top of the file.