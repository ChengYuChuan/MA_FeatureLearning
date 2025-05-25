# 🧠 Encoder Pre-training 
## 3D Autoencoder for Voxel-based Cube Reconstruction

This repository provides a PyTorch-based framework for training and evaluating a 3D convolutional autoencoder designed for reconstructing volumetric image cubes (e.g., from biomedical or material imaging modalities). The architecture leverages 3D UNet-inspired designs with residual blocks, advanced loss functions, and flexible preprocessing pipelines.

---

## 📁 Project Structure

```
.
├── train.py                 # Main training entry point
├── CubeDataset.py           # Custom dataset and DataLoader management
├── transform.py             # Data preprocessing and augmentation
├── buildingblocks.py        # Model components: encoders, decoders, residual blocks
├── loss.py                  # MSE, SSIM, hybrid loss implementations
├── utils.py                 # Checkpointing, logging, optimizers
├── AutoencoderTrainer.py    # Training manager with TensorBoard support
├── 3DUnet_py311.sh          # SLURM batch script for cluster training
└── environment.yml          # Conda environment specification
```

---

## 📦 Installation

To set up the environment using the provided Conda spec:

```bash
conda env create -f environment.yml
conda activate MAenv
```

The environment includes:

* Python 3.11
* PyTorch 2.5.1 with CUDA 12.4
* torchvision, numpy, scipy, tqdm
* GPU-ready setup with full CUDA stack

---

## 🏗️ Model Architecture

* **Encoder**: Stack of `ResBlockPNI` units with optional pooling.
* **Decoder**: Symmetric decoder with upsampling and skip connections.
* **Final Layer**: 1×1×1 3D convolution for voxel-wise reconstruction.
* **Basic Block**: Configurable with GroupNorm, ELU, and residual links.
* **Input/Output**: 5D tensors shaped `(N, C, D, H, W)` with cube sizes of 24 or 32.

---

## 🧪 Running Training

### 🛠️ SLURM (example)

```bash
sbatch 3DUnet_py311.sh
```

### 🧪 CLI Mode (standalone)

```bash
python train.py MSELoss Cubes 32 max 0.0002 5 0.5 false
```

**Arguments:**

| Position | Name           | Description                                                  |
|----------|----------------|--------------------------------------------------------------|
| 1        | `LossType`     | e.g., `MSELoss`, `SSIMLoss`, `L1SSIMLoss`, `HybridL1MSELoss` |
| 2        | `DataFolder`   | Folder name, e.g., `Cubes32`, `MaskedCubes32`                |
| 3        | `PoolType`     | `'avg'` or `'max'`                                           |
| 4        | `LearningRate` | e.g., `0.0002`                                               |

There are some of them are optional because you won't use those features all the time.
Only if you involve those parameters, you would need to pay attention of them.

---

## 🧠 Data Format

* Input data should be `.npy` files with shape `(CubeSize, CubeSize, CubeSize)`.
* Files are auto-split into training, validation, and test sets (80/10/10).
* Custom augmentations: flipping, rotation, Z-score + min-max normalization.

---

## 📉 Loss Functions

Supported loss types (`--LossType`):

| Loss Name         | Description                |
| ----------------- | -------------------------- |
| `MSELoss`         | Mean squared error         |
| `L1Loss`          | Mean absolute error        |
| `SSIMLoss`        | Structural similarity (3D) |
| `L1SSIMLoss`      | L1 + SSIM combined loss    |
| `HybridL1MSELoss` | L1 and MSE hybrid loss     |
| `MSESSIMLoss`     | MSE + SSIM combined        |
| `MS_SSIMLoss`     | Multi-scale 3D SSIM        |

The `loss.py` module provides extensive flexibility and composability.

---

## 📊 Logging and Checkpoints

* TensorBoard logs are stored in `CheckPoint_*/logs/`.
* Best model checkpoints saved as `best_checkpoint.pytorch`.
* Use TensorBoard for visual monitoring:

```bash
tensorboard --logdir=CheckPoint_BS1_RBPNI_32_4Layers_CD_Cube32_MSELoss_LR2e-4/logs
```

Includes:

* Input / Prediction image slices
* Encoder feature maps
* Loss and evaluation score curves

---

## 🧪 Example Dataset Folder Structure

[Download Cubes32.zip]()
```
Cubes32/
├── sample_001.npy  # shape: (32, 32, 32)
├── sample_002.npy
...
```
