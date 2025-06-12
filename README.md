# Feature Learning in 3D Voxel Data


## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Table of environment variables](#table-of-environment-variables)
- [Repository structure](#repository-structure)
- [Pipeline Overview](#Pipeline-Overview)
- [References](#references)
- [License](#license)

# Setup

## Requirements

## Setting up the environment
### Conda:
LAP: `NEW_CUNet.yml`

Autoencoder: `environment_GUNet.yml`
```sh
conda env create -f requirements_NEW_CUNet.yml
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
pip install -r requirements_NEW_CUNet.txt
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

## Autoencoder
The intention of training autoencoder is pre-training the encoder without the MLP part.

## LAP
it will show how pair-matching works.

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
├── GUNet
│   ├── CubeLAP.sh
│   ├── CubeLAP_main.py
│   ├── CubeLAPwMLP_main.py
│   ├── Data_Stats
│   │   ├── Data_Stats.txt
│   │   └── stats.py
│   ├── ENV_files
│   │   ├── environment_GUNet.yml
│   │   ├── requirements_GUNet.txt
│   │   ├── requirements_NEW_CUNet.txt
│   │   └── requirements_NEW_CUNet.yml
│   ├── GUNet.sh
│   ├── TensorBoard
│   ├── logs
│   ├── pretrain_encoder_main.py
│   └── src_GUNet
│       ├── __init__.py
│       ├── __pycache__
│       │   └── __init__.cpython-38.pyc
│       ├── architectures
│       │   ├── FeatureEncoder.py
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   ├── decoder.py
│       │   ├── dilated_dense.py
│       │   ├── encoder.py
│       │   └── unet.py
│       ├── groups
│       │   ├── S4_group.py
│       │   ├── T4_group.py
│       │   ├── V_group.py
│       │   ├── __init__.py
│       │   └── __pycache__
│       ├── layers
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   ├── convs.py
│       │   └── gconvs.py
│       ├── training
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   ├── datamodule.py
│       │   ├── datamodule_LAP.py
│       │   ├── lightningLAPNet.py
│       │   ├── lightningLAPNetwMLP.py
│       │   ├── lightningUnet.py
│       │   └── loss.py
│       └── utils
│           ├── CheckPoint
│           ├── __init__.py
│           ├── __pycache__
│           ├── concatenation
│           ├── dropout
│           ├── helpers
│           ├── interpolation
│           ├── logging
│           ├── normalization
│           ├── plots
│           └── pooling
├── LICENSE
└── README.md
```

# Pipeline Overview

---

## **Step 0: `Step0.py`**

Extracts 3D cubes (e.g., 32×32×32) centered on annotated neuron coordinates. Combines raw and masked voxel channels.

* **Input**: `.raw` volumes + `.txt` annotation files
* **Output**: 2-channel `.npy` cubes
* **Key features**: Robust padding, coordinate alignment, optional `.raw` or `.npy` output
* Key Reference: `Coordinate_of_Cell`
* Download: 
  * [Crop Raw](https://drive.google.com/file/d/1GUNUqphCDl55A1IlVHcDgbF0k1Ndpqou/view?usp=drive_link)
  * [Masked](https://drive.google.com/file/d/1tO4FY32HbxaplBzLRTMPqHRc7pMYPE_r/view?usp=drive_link)

NOTE: you can only crop raw Cube from the original worm data. (Skip the concatenation of Mask section)

NOTE: If you only need original data (without mask), you can just skip Step1.py and Step2.py (directly run Merge.py right after Step0.py)

---

## **Step 1: `Step1.py`**

Adds a **soft boundary mask** (third channel) using a sigmoid-transformed distance from labeled regions.

* **Input**: 2-channel cubes
* **Output**: 3-channel cubes (`[raw, binary_mask, soft_mask]`)
* **Method**: Applies a decaying weight outside label boundaries

---

## **Step 2: `Step2.py`**

Extracts only the soft-masked voxel (`3rd channel`) for direct use in model training.

* **Input**: 3-channel cubes
* **Output**: 1-channel soft-masked `.npy` cubes
* **Purpose**: Simplifies dataset for downstream models

---

## **Merge: `Merge.py`**

Merges all individual cubes per worm into a single 5D array for efficient batched access.

* **Shape**: `(558, 1, 32, 32, 32)` per worm
* **Output**: One `.npy` file per worm
* **Note**: Automatically fills missing cells with zeroed cubes

---

## **Validation: `TestSize.py`**

Scans the merged dataset and identifies any `.npy` files with incorrect shapes or loading issues.

* **Output**: List of problematic files, if any

---

## **Visualization: `VisMergedData.py`**

Visualizes 3 random cells from a random worm, displaying:

* XY, XZ, YZ mid-slices

* Interactive colormap support (`viridis`, `inferno`, etc.)

* **Output**: Interactive matplotlib plot

---

## ✅ Recommended Workflow

```bash
# 1. Extract 2-channel cubes from raw + mask volumes
python Step0.py

# 2. Add soft boundary as 3rd channel
python Step1.py

# 3. Extract soft-masked cube for training
python Step2.py

# 4. Merge all cubes per worm
python Merge.py

# 5. Check for invalid files
python TestSize.py

# 6. Visualize samples
python VisMergedData.py
```

---

## 📦 Output Directory Structure

```
├── 2ChannelMaskedCube32/      # After Step0
├── 3ChannelMaskedCube32/      # After Step1
├── MaskedCube32/              # After Step2
├── MergedCubes32/             # After Merge
├── skipped_files.txt          # From Step1
├── processed_files.txt        # From Step1
```
---

# Some Explanation of other Reference files or folders

## 🧬 `Coordinates of cube.zip`

### 🔹 Folder: `Coordinate_of_Cell/`

This directory contains `.txt` files with **annotated neuron coordinates** for each worm, used as inputs in `Step0.py`.

### 📄 File Format: `worm_###.txt`

Each file corresponds to a single worm and contains tab-separated information about its identified neurons. The key columns include:

| Column         | Description                                                |
| -------------- | ---------------------------------------------------------- |
| `label_number` | Unique numeric ID for the neuron                           |
| `label_name`   | Neuron name (e.g., `ADAL`, `PHAR`, `RIGL`, etc.)           |
| `x`, `y`, `z`  | 3D coordinates of the neuron center (float precision)      |
| `Aligned_No`   | A unique sequential index used as the identifier in output |

### 🔍 Example Row:

```
label_number	label_name	    x	            y	          z	        Aligned_No
...
195	        ADAL	            1242.000000	    84.000000	  33.000000	1
...
```

* `ADAL` is the neuron label.
* The position is at `(x=1242, y=84, z=33)`.
* This is the first aligned neuron in the list.

### 🧪 Usage in Step0

In `Step0.py`, these files are loaded via:

```python
txt_path = os.path.join(txt_dir, txt_file)
coordinates = read_all_coordinates(txt_path)
```

Each row provides a target for extracting a **32×32×32 cube** around the neuron center. The extracted cube is named as:

```
<worm_name>_cube_<Aligned_No>.npy
```


# References

Part of this repository was taken from the [Cubenet repository](https://github.com/danielewworrall/cubenet), which implements some model examples described in this [ECCV18 article](https://arxiv.org/abs/1804.04458):

```
@inproceedings{Worrall18,
  title     = {CubeNet: Equivariance to 3D Rotation and Translation},
  author    = {Daniel E. Worrall and Gabriel J. Brostow},
  booktitle = {Computer Vision - {ECCV} 2018 - 15th European Conference, Munich,
               Germany, September 8-14, 2018, Proceedings, Part {V}},
  pages     = {585--602},
  year      = {2018},
  doi       = {10.1007/978-3-030-01228-1\_35},
}
```

The code in `./src_GUNet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py), which corresponds to:

```
@article{SwitchableNorm,
  title={Differentiable Learning-to-Normalize via Switchable Normalization},
  author={Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li},
  journal={International Conference on Learning Representation (ICLR)},
  year={2019}
}
```

Some of the code in `./src_GUNet/architectures` was inspired from this [3D U-Net repository](https://github.com/JielongZ/3D-UNet-PyTorch-Implementation), as well as from the structure described in [Dilated Dense U-Net for Infant Hippocampus Subfield Segmentation](https://www.frontiersin.org/articles/10.3389/fninf.2019.00030/full):

```
@article{zhu_dilated_2019,
	title = {Dilated Dense U-Net for Infant Hippocampus Subfield Segmentation},
	url = {https://www.frontiersin.org/article/10.3389/fninf.2019.00030/full},
	doi = {10.3389/fninf.2019.00030},
	journaltitle = {Front. Neuroinform.},
	author = {Zhu, Hancan and Shi, Feng and Wang, Li and Hung, Sheng-Che and Chen, Meng-Hsiang and Wang, Shuai and Lin, Weili and Shen, Dinggang},
	year = {2019},
}
```

Some of the code for the losses in `./src_GUNet/training` was taken from this [Repository: Differentiation of Blackbox Combinatorial Solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers), which corresponds to:

```
@inproceedings{VlastelicaEtal2020:BBoxSolvers,
  title = {Differentiation of Blackbox Combinatorial Solvers},
  booktitle = {International Conference on Learning Representations},
  series = {ICLR'20},
  month = may,
  year = {2020},
  note = {*Equal Contribution},
  slug = {vlastelicaetal2020-bboxsolvers},
  author = {Vlastelica*, Marin and Paulus*, Anselm and Musil, V{\'i}t and Martius, Georg and Rol{\'i}nek, Michal},
  url = {https://openreview.net/forum?id=BkevoJSYPB},
  month_numeric = {5}
}
```

# License

This repository is covered by the MIT license, but some exceptions apply, and are listed below:
- The file in `./src_GUNet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py) by Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li, and is covered by the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/), as mentionned also at the top of the file.