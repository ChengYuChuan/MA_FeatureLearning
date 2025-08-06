# Feature Learning in 3D Voxel Data


## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Environment Variables Table](#Environment-Variables-Table)
- [Repository structure](#repository-structure)
- [License](#license)

# Setup
## Setting up the environment
### Conda:
LAP: `NEW_CUNet.yml`

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

---

## Usage

This project supports node correspondence learning based on Graph Neural Networks (GNNs), using the PyTorch Lightning framework for modular training and management.

### 1. Prepare Graph Data

First, use `Graph_building.py` to generate `.pt` files containing PyG `Data` objects. These graph files should be stored in a specified folder (e.g., `./Graph_Data/DoubleConv/Graph_data_MLP512_8_3Layers_L1_R45`).

### 2. Configure Parameters

You can modify `Dynamic_main8.py` or `Dynamic_main16.py` to customize the training process. Supported options include:

* Use a fixed lambda value or enable dynamic (annealing) lambda scheduling for the loss function.
* Choose a distance metric for the loss, such as `MSE` or `L1`.
* Adjust the GNN architecture (e.g., hidden dimension or number of GAT heads).
* Configure training settings such as number of epochs, batch size, dropout rate, learning rate, etc.

We recommend centralizing these settings in a `.env` file and loading them with `python-dotenv` for better configuration management (not yet implemented but easy to extend).

### 3. Run the Training Script

Use the following commands to start training:

```bash
python Dynamic_main8.py
```

or

```bash
python Dynamic_main16.py
```

These two scripts correspond to different node feature configurations (e.g., 8 or 16 layer MLP embeddings).




# Environment Variables Table
## 2. Dataset Settings

| Variable Name   | Description                                                                                             | Suggested Default |
|----------------|---------------------------------------------------------------------------------------------------------|-------------------|
| `PATH_TO_DATA` | Path to the folder containing the dataset.                                                              | `./data`          |
| `GRAPH_DATA_FOLDER` | Path to the folder that contains the graph data.                                                        | `./Graph_Data/DoubleConv/Graph_data_MLP512_16_3Layers_L1_R45` |
| `BATCH_SIZE`   | Batch size for the dataloader.                                                                          | `4`               |
| `NUM_WORKERS`  | Number of CPU workers for the dataloader.                                                               | `2`               |

## 3. Model & Group Settings

| Variable Name     | Description                                                                                           | Suggested Default |
|------------------|-------------------------------------------------------------------------------------------------------|-------------------|
| `NODE_FEATURE_DIM` | Input dimension of the node feature vector. `8` for only Geo features, `520` for full features.      | `520`             |
| `GNN_HIDDEN_DIM` | Hidden layer dimension in GNN.                                                                         | `256`             |
| `GNN_OUTPUT_DIM` | Output dimension of GNN encoder.                                                                       | `128`             |
| `GAT_HEADS`      | Number of attention heads for GAT (Graph Attention Network).                                          | `8`               |
| `DROPOUT`        | Dropout rate.                                                                                          | `0.0`             |

## 4. Logs & Saving

| Variable Name | Description                                                                                             | Suggested Default |
|---------------|---------------------------------------------------------------------------------------------------------|-------------------|
| `LOG_NAME`    | Name prefix for this specific run in Tensorboard and results folders.                                   | `MLP512_16_3Layers_L1_R45_200EpochES_Lambda200.00_LR0.0005` |

## 5. Loss Function & Optimizer

| Variable Name     | Description                                                                                          | Suggested Default |
|------------------|------------------------------------------------------------------------------------------------------|-------------------|
| `LEARNING_RATE`  | The learning rate for the optimizer.                                                                 | `0.0005`          |
| `DISTANCE_TYPE`  | The distance metric used for the loss function.                                                      | `MSE`             |
| `LAMBDA`         | Fixed lambda value. Used when lambda scheduler is disabled.                                          | `200.0`           |

## 6. Trainer Settings

| Variable Name               | Description                                                                                 | Suggested Default |
|----------------------------|---------------------------------------------------------------------------------------------|-------------------|
| `MAX_EPOCHS`               | Maximum number of epochs to train for.                                                      | `1000`            |
| `LOG_EVERY_N_STEPS`        | How often to log metrics every N steps.                                                     | `50`              |
| `PROGRESS_BAR_REFRESH_RATE`| Refresh rate for the progress bar.                                                          | `50`              |

## 8. Lambda Scheduling (Conditional)

| Variable Name         | Description                                                                                  | Suggested Default |
|----------------------|----------------------------------------------------------------------------------------------|-------------------|
| `USE_LAMBDA_SCHEDULER` | Whether to use dynamic lambda scheduling (annealing).                                       | `False`           |
| `LAMBDA_START`       | Starting value of lambda if using annealing.                                                 | `50.0`            |
| `LAMBDA_END`         | Ending value of lambda if using annealing.                                                   | `200.0`           |
| `LAMBDA_WARMUP_EPOCHS`| Number of epochs to warm-up the lambda value during annealing.                              | `10`              |


# Repository structure

```sh
.
├── CheckPoints
│   ├── DoubleConv
│   │   ├── MLP512_16_3Layers_L1_R45
│   │   │   ├── checkpoints_MLP512_16_3Layers_L1_R45_100EpochES_Lambda200.00_LR0.0002
│   │   │   │   ├── graph-matcher-epoch=315-val_acc=0.6063.ckpt
│   │   │   │   ├── graph-matcher-epoch=316-val_acc=0.6267.ckpt
│   │   │   │   └── graph-matcher-epoch=342-val_acc=0.6075.ckpt
│   │   │   ├── checkpoints_MLP512_16_3Layers_L1_R45_100EpochES_Lambda200.00_LR0.0005
│   │   │   │   ├── graph-matcher-epoch=566-val_acc=0.6781.ckpt
│   │   │   │   ├── graph-matcher-epoch=612-val_acc=0.6711.ckpt
│   │   │   │   └── graph-matcher-epoch=625-val_acc=0.6661.ckpt
│   │   │   └── checkpoints_MLP512_16_3Layers_L1_R45_200EpochES_Lambda200.00_LR0.0005
│   │   │       ├── graph-matcher-epoch=315-val_acc=0.5244.ckpt
│   │   │       ├── graph-matcher-epoch=316-val_acc=0.5263.ckpt
│   │   │       └── graph-matcher-epoch=340-val_acc=0.5224.ckpt
│   │   ├── MLP512_16_3Layers_MSE_R45
│   │   │   ├── checkpoints_MLP512_16_3Layers_MSE_R45_100EpochES_Lambda200.00_LR0.0002
│   │   │   │   ├── graph-matcher-epoch=229-val_acc=0.5638.ckpt
│   │   │   │   ├── graph-matcher-epoch=315-val_acc=0.5432.ckpt
│   │   │   │   └── graph-matcher-epoch=316-val_acc=0.5547.ckpt
│   │   │   ├── checkpoints_MLP512_16_3Layers_MSE_R45_100EpochES_Lambda200.00_LR0.0005
│   │   │   │   ├── graph-matcher-epoch=316-val_acc=0.6174.ckpt
│   │   │   │   ├── graph-matcher-epoch=363-val_acc=0.6281.ckpt
│   │   │   │   └── graph-matcher-epoch=388-val_acc=0.6398.ckpt
│   │   │   └── checkpoints_MLP512_16_3Layers_MSE_R45_200EpochES_Lambda200.00_LR0.0005
│   │   │       ├── graph-matcher-epoch=388-val_acc=0.6023.ckpt
│   │   │       ├── graph-matcher-epoch=393-val_acc=0.5624.ckpt
│   │   │       └── graph-matcher-epoch=394-val_acc=0.5618.ckpt
│   │   ├── MLP512_16_4Layers_L1_R45
│   │   │   └── checkpoints_MLP512_16_4Layers_L1_R45_100EpochES_Lambda200.00_LR0.0002
│   │   │       ├── graph-matcher-epoch=486-val_acc=0.4781.ckpt
│   │   │       ├── graph-matcher-epoch=512-val_acc=0.4783.ckpt
│   │   │       └── graph-matcher-epoch=566-val_acc=0.4975.ckpt
│   │   ├── MLP512_16_4Layers_MSE_R45
│   │   │   └── checkpoints_MLP512_16_4Layers_MSE_R45_100EpochES_Lambda200.00_LR0.0002
│   │   │       ├── graph-matcher-epoch=126-val_acc=0.5116.ckpt
│   │   │       ├── graph-matcher-epoch=205-val_acc=0.4878.ckpt
│   │   │       └── graph-matcher-epoch=206-val_acc=0.4885.ckpt
│   │   ├── MLP512_8_3Layers_L1_R45
│   │   │   ├── checkpoints_MLP512_8_3Layers_L1_R45_100EpochES_Lambda100.00_LR0.00020
│   │   │   │   ├── graph-matcher-epoch=316-val_acc=0.5685.ckpt
│   │   │   │   ├── graph-matcher-epoch=316-val_acc=0.5706.ckpt
│   │   │   │   └── graph-matcher-epoch=337-val_acc=0.5622.ckpt
│   │   │   ├── checkpoints_MLP512_8_3Layers_L1_R45_100EpochES_Lambda100.00_LR0.00050
│   │   │   │   ├── graph-matcher-epoch=214-val_acc=0.5498.ckpt
│   │   │   │   ├── graph-matcher-epoch=229-val_acc=0.5642.ckpt
│   │   │   │   └── graph-matcher-epoch=242-val_acc=0.5346.ckpt
│   │   │   ├── checkpoints_MLP512_8_3Layers_L1_R45_100EpochES_Lambda200.00_LR0.00020
│   │   │   │   ├── graph-matcher-epoch=316-val_acc=0.5690.ckpt
│   │   │   │   ├── graph-matcher-epoch=363-val_acc=0.5711.ckpt
│   │   │   │   └── graph-matcher-epoch=367-val_acc=0.5573.ckpt
│   │   │   └── checkpoints_MLP512_8_3Layers_L1_R45_200EpochES_Lambda200.00_LR0.00020
│   │   │       ├── graph-matcher-epoch=204-val_acc=0.5140.ckpt
│   │   │       ├── graph-matcher-epoch=214-val_acc=0.5133.ckpt
│   │   │       └── graph-matcher-epoch=252-val_acc=0.4995.ckpt
│   │   ├── MLP512_8_3Layers_MSE_R45
│   │   │   ├── checkpoints_MLP512_8_3Layers_MSE_R45_100EpochES_Lambda200.00_LR0.00020
│   │   │   │   ├── graph-matcher-epoch=214-val_acc=0.6192.ckpt
│   │   │   │   ├── graph-matcher-epoch=229-val_acc=0.6217.ckpt
│   │   │   │   └── graph-matcher-epoch=252-val_acc=0.6068.ckpt
│   │   │   ├── checkpoints_MLP512_8_3Layers_MSE_R45_100EpochES_Lambda200.00_LR0.00050
│   │   │   │   ├── graph-matcher-epoch=229-val_acc=0.5392.ckpt
│   │   │   │   ├── graph-matcher-epoch=252-val_acc=0.5441.ckpt
│   │   │   │   └── graph-matcher-epoch=262-val_acc=0.5435.ckpt
│   │   │   └── checkpoints_MLP512_8_3Layers_MSE_R45_200EpochES_Lambda200.00_LR0.00020
│   │   │       ├── graph-matcher-epoch=134-val_acc=0.6081.ckpt
│   │   │       ├── graph-matcher-epoch=214-val_acc=0.6068.ckpt
│   │   │       └── graph-matcher-epoch=229-val_acc=0.6127.ckpt
│   │   ├── MLP512_8_4Layers_L1_R45
│   │   │   └── checkpoints_MLP512_8_4Layers_L1_R45_100EpochES_Lambda200.00_LR0.00020
│   │   │       ├── graph-matcher-epoch=262-val_acc=0.5464.ckpt
│   │   │       ├── graph-matcher-epoch=284-val_acc=0.5301.ckpt
│   │   │       └── graph-matcher-epoch=316-val_acc=0.5360.ckpt
│   │   └── MLP512_8_4Layers_MSE_R45
│   │       └── checkpoints_MLP512_8_4Layers_MSE_R45_100EpochES_Lambda200.00_LR0.00020
│   │           ├── graph-matcher-epoch=388-val_acc=0.4471.ckpt
│   │           ├── graph-matcher-epoch=393-val_acc=0.4238.ckpt
│   │           └── graph-matcher-epoch=408-val_acc=0.4185.ckpt
│   ├── Only_Geo
│   │   ├── 438289_Geo_R30.txt
│   │   ├── 438290_Geo_R45.txt
│   │   ├── 438291_Geo_R60.txt
│   │   ├── checkpoints_geo_features_R30_100EpochES_Lambda50.00_LR0.0005
│   │   │   ├── graph-matcher-epoch=151-val_acc=0.44.ckpt
│   │   │   ├── graph-matcher-epoch=176-val_acc=0.42.ckpt
│   │   │   └── graph-matcher-epoch=186-val_acc=0.43.ckpt
│   │   ├── checkpoints_geo_features_R35_100EpochES_Lambda50.00_LR0.0005
│   │   │   ├── graph-matcher-epoch=151-val_acc=0.5043.ckpt
│   │   │   ├── graph-matcher-epoch=167-val_acc=0.5188.ckpt
│   │   │   └── graph-matcher-epoch=176-val_acc=0.4676.ckpt
│   │   ├── checkpoints_geo_features_R40_100EpochES_Lambda50.00_LR0.0005
│   │   │   ├── graph-matcher-epoch=151-val_acc=0.5111.ckpt
│   │   │   ├── graph-matcher-epoch=167-val_acc=0.5029.ckpt
│   │   │   └── graph-matcher-epoch=176-val_acc=0.4815.ckpt
│   │   ├── checkpoints_geo_features_R45_100EpochES_Lambda50.00_LR0.0005
│   │   │   ├── graph-matcher-epoch=126-val_acc=0.48.ckpt
│   │   │   ├── graph-matcher-epoch=151-val_acc=0.53.ckpt
│   │   │   └── graph-matcher-epoch=167-val_acc=0.49.ckpt
│   │   ├── checkpoints_geo_features_R50_100EpochES_Lambda50.00_LR0.0005
│   │   │   ├── graph-matcher-epoch=126-val_acc=0.4857.ckpt
│   │   │   ├── graph-matcher-epoch=151-val_acc=0.5254.ckpt
│   │   │   └── graph-matcher-epoch=207-val_acc=0.4796.ckpt
│   │   ├── checkpoints_geo_features_R55_100EpochES_Lambda50.00_LR0.0005
│   │   │   ├── graph-matcher-epoch=151-val_acc=0.5407.ckpt
│   │   │   ├── graph-matcher-epoch=186-val_acc=0.5027.ckpt
│   │   │   └── graph-matcher-epoch=207-val_acc=0.5385.ckpt
│   │   └── checkpoints_geo_features_R60_100EpochES_Lambda50.00_LR0.0005
│   │       ├── graph-matcher-epoch=151-val_acc=0.55.ckpt
│   │       ├── graph-matcher-epoch=167-val_acc=0.52.ckpt
│   │       └── graph-matcher-epoch=207-val_acc=0.54.ckpt
│   └── RBPI
│       ├── MLP512_16_3Layers_L1_R45
│       │   ├── checkpoints_MLP512_16_3Layers_L1_R45_100EpochES_Lambda200.00_LR0.0001
│       │   │   ├── graph-matcher-epoch=316-val_acc=0.6765.ckpt
│       │   │   ├── graph-matcher-epoch=363-val_acc=0.6670.ckpt
│       │   │   └── graph-matcher-epoch=367-val_acc=0.6715.ckpt
│       │   ├── checkpoints_MLP512_16_3Layers_L1_R45_100EpochES_Lambda200.00_LR0.0002
│       │   │   ├── graph-matcher-epoch=316-val_acc=0.6724.ckpt
│       │   │   ├── graph-matcher-epoch=363-val_acc=0.6674.ckpt
│       │   │   └── graph-matcher-epoch=433-val_acc=0.6629.ckpt
│       │   ├── checkpoints_MLP512_16_3Layers_L1_R45_100EpochES_Lambda50.00_LR0.0008
│       │   │   ├── graph-matcher-epoch=340-val_acc=0.6588.ckpt
│       │   │   ├── graph-matcher-epoch=342-val_acc=0.6647.ckpt
│       │   │   └── graph-matcher-epoch=363-val_acc=0.6640.ckpt
│       │   └── checkpoints_MLP512_16_3Layers_L1_R45_200EpochES_Lambda200.00_LR0.0002
│       │       ├── graph-matcher-epoch=724-val_acc=0.7375.ckpt
│       │       ├── graph-matcher-epoch=822-val_acc=0.7409.ckpt
│       │       └── graph-matcher-epoch=836-val_acc=0.7369.ckpt
│       ├── MLP512_16_3Layers_MSE_R45
│       │   ├── checkpoints_MLP512_16_3Layers_MSE_R45_100EpochES_Lambda200.00_LR0.0002
│       │   │   ├── graph-matcher-epoch=522-val_acc=0.6991.ckpt
│       │   │   ├── graph-matcher-epoch=566-val_acc=0.7027.ckpt
│       │   │   └── graph-matcher-epoch=579-val_acc=0.6982.ckpt
│       │   ├── checkpoints_MLP512_16_3Layers_MSE_R45_100EpochES_Lambda50.00_LR0.0004
│       │   │   ├── graph-matcher-epoch=433-val_acc=0.6823.ckpt
│       │   │   ├── graph-matcher-epoch=458-val_acc=0.6780.ckpt
│       │   │   └── graph-matcher-epoch=509-val_acc=0.6853.ckpt
│       │   └── checkpoints_MLP512_16_3Layers_MSE_R45_200EpochES_Lambda200.00_LR0.0002
│       │       ├── graph-matcher-epoch=689-val_acc=0.7276.ckpt
│       │       ├── graph-matcher-epoch=847-val_acc=0.7262.ckpt
│       │       └── graph-matcher-epoch=874-val_acc=0.7274.ckpt
│       ├── MLP512_16_4Layers_L1_R45
│       │   └── checkpoints_MLP512_16_4Layers_L1_R45_100EpochES_Lambda200.00_LR0.0002
│       │       ├── graph-matcher-epoch=625-val_acc=0.6477.ckpt
│       │       ├── graph-matcher-epoch=655-val_acc=0.6461.ckpt
│       │       └── graph-matcher-epoch=670-val_acc=0.6573.ckpt
│       ├── MLP512_16_4Layers_MSE_R45
│       │   ├── checkpoints_MLP512_16_4Layers_MSE_R45_100EpochES_Lambda200.00_LR0.0002
│       │   │   ├── graph-matcher-epoch=229-val_acc=0.6428.ckpt
│       │   │   ├── graph-matcher-epoch=252-val_acc=0.6434.ckpt
│       │   │   └── graph-matcher-epoch=342-val_acc=0.6339.ckpt
│       │   └── checkpoints_MLP512_16_4Layers_MSE_R45_100EpochES_Lambda200.00_LR0.0005
│       │       ├── graph-matcher-epoch=104-val_acc=0.5812.ckpt
│       │       ├── graph-matcher-epoch=112-val_acc=0.5740.ckpt
│       │       └── graph-matcher-epoch=97-val_acc=0.5710.ckpt
│       ├── MLP512_8_3Layers_L1_R45
│       │   ├── checkpoints_MLP512_8_3Layers_L1_R45_100EpochES_Lambda200.00_LR0.00020
│       │   │   ├── graph-matcher-epoch=229-val_acc=0.6384.ckpt
│       │   │   ├── graph-matcher-epoch=229-val_acc=0.6452.ckpt
│       │   │   └── graph-matcher-epoch=242-val_acc=0.6206.ckpt
│       │   └── checkpoints_MLP512_8_3Layers_L1_R45_200EpochES_Lambda200.00_LR0.00020
│       │       ├── graph-matcher-epoch=229-val_acc=0.6324.ckpt
│       │       ├── graph-matcher-epoch=242-val_acc=0.6081.ckpt
│       │       └── graph-matcher-epoch=252-val_acc=0.6188.ckpt
│       ├── MLP512_8_3Layers_MSE_R45
│       │   ├── checkpoints_MLP512_8_3Layers_MSE_R45_100EpochES_Lambda200.00_LR0.00020
│       │   │   ├── graph-matcher-epoch=126-val_acc=0.7217.ckpt
│       │   │   ├── graph-matcher-epoch=200-val_acc=0.7142.ckpt
│       │   │   └── graph-matcher-epoch=214-val_acc=0.7237.ckpt
│       │   └── checkpoints_MLP512_8_3Layers_MSE_R45_200EpochES_Lambda200.00_LR0.00020
│       │       ├── graph-matcher-epoch=126-val_acc=0.7258.ckpt
│       │       ├── graph-matcher-epoch=214-val_acc=0.7179.ckpt
│       │       └── graph-matcher-epoch=220-val_acc=0.7211.ckpt
│       ├── MLP512_8_4Layers_L1_R45
│       │   └── checkpoints_MLP512_8_4Layers_L1_R45_100EpochES_Lambda200.00_LR0.00020
│       │       ├── graph-matcher-epoch=521-val_acc=0.6306.ckpt
│       │       ├── graph-matcher-epoch=523-val_acc=0.6263.ckpt
│       │       └── graph-matcher-epoch=620-val_acc=0.6306.ckpt
│       └── MLP512_8_4Layers_MSE_R45
│           ├── checkpoints_MLP512_8_4Layers_MSE_R45_100EpochES_Lambda0.50_LR0.00005
│           │   ├── graph-matcher-epoch=316-val_acc=0.5846.ckpt
│           │   ├── graph-matcher-epoch=388-val_acc=0.5878.ckpt
│           │   └── graph-matcher-epoch=458-val_acc=0.5833.ckpt
│           └── checkpoints_MLP512_8_4Layers_MSE_R45_100EpochES_Lambda200.00_LR0.00020
│               ├── graph-matcher-epoch=342-val_acc=0.5869.ckpt
│               ├── graph-matcher-epoch=363-val_acc=0.5887.ckpt
│               └── graph-matcher-epoch=367-val_acc=0.5964.ckpt
├── Dynamic_main16.py
├── Dynamic_main8.py
├── GNN_LAP_main16.sh
├── GNN_LAP_main8.sh
├── GNN_buildGraph.sh
├── GNN_stats.sh
├── Graph_building.py
├── loss.py
├── requirements.txt
├── setup_env.txt

62 directories, 138 files

```

# License

This repository is covered by the MIT license, but some exceptions apply, and are listed below:
- The file in `./src_GUNet/utils/normalization/SwitchNorm3d` was taken from the [SwitchNorm repository](https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py) by Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li, and is covered by the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/), as mentionned also at the top of the file.