import os
import random
import logging
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from loss import DifferentiableHungarianLoss, compute_distance_matrix

from pytorch_lightning.callbacks.progress import TQDMProgressBar

# ====================
# Logging 設定 (重複，但為了每個文件獨立運行)
# ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("main_training.log"),  # ### MODIFIED ###
        logging.StreamHandler()
    ]
)


# ===================================================================
# 1. 自訂 Collate 函數 (這是解決 TypeError 的關鍵)
# ===================================================================
def graph_pair_collate(batch):
    """
    自訂的 collate 函數，用於處理 PyG Data 物件的配對。
    DataLoader 會將一個樣本包裹在一個列表中，例如 [(graph1, graph2)]。
    這個函數只是簡單地將其解包，返回 (graph1, graph2)。
    """
    return batch[0]


# ===================================================================
# 2. 核心模型定義 (GNN Encoder)
# ===================================================================
class EmbeddingGAT(nn.Module):
    # ### MODIFIED ### 將 dropout_rate 作為參數傳入，默認0.1
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, consensus=False, dropout_rate=0.1):
        super().__init__()
        self.consensus = consensus
        self.dropout_rate = dropout_rate  # 保存 dropout_rate

        # ### MODIFIED ### 將 GATConv 內部的 dropout 設為 0
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.0)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        # ### MODIFIED ### 將 GATConv 內部的 dropout 設為 0
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.0)
        self.norm2 = nn.LayerNorm(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, graph_data: Data):
        x, edge_index = graph_data.x, graph_data.edge_index
        x_shortcut = self.shortcut(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        # ### MODIFIED ### 使用統一的 dropout_rate
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # ### MODIFIED ### 默認 consensus=False，但保留邏輯
        if self.consensus:
            N_nodes = x.shape[0]
            values = torch.ones(edge_index.shape[1], device=x.device)
            A_sparse = torch.sparse_coo_tensor(edge_index, values, (N_nodes, N_nodes))
            enhanced_x = torch.sparse.mm(A_sparse, x)
            x = x + enhanced_x

        x = self.conv2(x, edge_index)
        x = self.norm2(x + x_shortcut)
        x = F.elu(x)

        return x


# ===================================================================
# 3. 資料模組 (LightningDataModule) - 動態配對策略
# ===================================================================
class DynamicGraphPairDataset(Dataset):
    def __init__(self, graph_files):
        self.graph_files = graph_files
        self.num_graphs = len(self.graph_files)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        path1 = self.graph_files[idx]
        rand_idx = random.randint(0, self.num_graphs - 1)
        while rand_idx == idx:
            rand_idx = random.randint(0, self.num_graphs - 1)
        path2 = self.graph_files[rand_idx]

        graph1 = torch.load(path1)
        graph2 = torch.load(path2)
        return graph1, graph2


class GraphPairDataModule(pl.LightningDataModule):
    def __init__(self, graph_folder: str, batch_size: int = 1, val_split: float = 0.2, seed: int = 42,
                 num_workers: int = 0):
        super().__init__()
        self.graph_folder = Path(graph_folder)
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.num_workers = num_workers
        self.train_files = []
        self.val_files = []

    def setup(self, stage: str = None):
        graph_files = sorted([f for f in self.graph_folder.glob("*.pt")])
        if not graph_files:
            raise FileNotFoundError(f"在 {self.graph_folder} 中找不到任何 .pt 圖檔案。")

        rng = random.Random(self.seed)
        rng.shuffle(graph_files)

        split_idx = int(len(graph_files) * (1 - self.val_split))
        self.train_files = graph_files[:split_idx]
        self.val_files = graph_files[split_idx:]

        logging.info(f"資料集切分完成：訓練圖 {len(self.train_files)} 個，驗證圖 {len(self.val_files)} 個。")

    def train_dataloader(self):
        dataset = DynamicGraphPairDataset(self.train_files)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=graph_pair_collate
        )

    def val_dataloader(self):
        dataset = DynamicGraphPairDataset(self.val_files)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=graph_pair_collate
        )


# ===================================================================
# 4. 核心訓練模組 (LightningModule)
# ===================================================================
class GraphMatcherLightning(pl.LightningModule):
    # ### MODIFIED ###: 加入 scheduler 相關參數，並提供預設值
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, criterion,
                 heads=4, dropout_rate=0.1,
                 # scheduler 參數
                 use_lambda_scheduler: bool = False,
                 lambda_start: float = None,
                 lambda_end: float = None,
                 lambda_warmup_epochs: int = None,
                 total_epochs: int = None):
        super().__init__()
        # 使用 save_hyperparameters 保存所有參數，方便追蹤和恢復
        self.save_hyperparameters(ignore=['criterion'])

        self.gnn_encoder = EmbeddingGAT(in_channels, hidden_channels, out_channels, heads, consensus=False,
                                        dropout_rate=dropout_rate)
        self.criterion = criterion

    # ### NEW ###: 加入 on_train_epoch_start hook，但只有在啟用時才執行
    def on_train_epoch_start(self):
        """在每個訓練 epoch 開始時，若啟用 scheduler，則計算並更新 lambda 值。"""
        # 只有當 use_lambda_scheduler 為 True 時才執行以下邏輯
        if not self.hparams.use_lambda_scheduler:
            return

        epoch = self.current_epoch
        hparams = self.hparams

        # 實作 Warmup + Linear Annealing 策略
        if epoch < hparams.lambda_warmup_epochs:
            current_lambda = hparams.lambda_start
        else:
            progress = (epoch - hparams.lambda_warmup_epochs) / (hparams.total_epochs - hparams.lambda_warmup_epochs)
            progress = min(progress, 1.0)
            current_lambda = hparams.lambda_start + progress * (hparams.lambda_end - hparams.lambda_start)

        # 更新 loss function 中的 lambda 值
        self.criterion.lambda_val = current_lambda

        # 記錄 lambda 值，方便監控
        self.log('lambda', current_lambda, on_step=False, on_epoch=True, prog_bar=True)

    def forward(self, graph_data):
        # ... forward 保持不變 ...
        return self.gnn_encoder(graph_data)

    def _common_step(self, batch, batch_idx):
        # ... _common_step 保持不變 ...
        graph1, graph2 = batch
        feats1 = self(graph1)
        feats2 = self(graph2)
        if feats1.shape[0] != feats2.shape[0]:
            logging.warning(f"警告：節點數不匹配，跳過此批次。Graph1: {feats1.shape[0]}, Graph2: {feats2.shape[0]}")
            return None, None
        latent_features = torch.stack([feats1, feats2], dim=0)
        inv_perm1 = torch.argsort(graph1.perm).to(self.device)
        inv_perm2 = torch.argsort(graph2.perm).to(self.device)
        loss, (row_ind, col_ind) = self.criterion(latent_features, inv_perm_A=inv_perm1, inv_perm_B=inv_perm2)
        acc = self._calculate_accuracy(row_ind, col_ind, inv_perm1, inv_perm2)
        return loss, acc

    # ... training_step, validation_step, configure_optimizers, _calculate_accuracy 保持不變 ...
    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        if loss is not None:
            current_batch_size = 1
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        if loss is not None:
            current_batch_size = 1
            self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=current_batch_size)
            self.log('val_acc', acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=current_batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def _calculate_accuracy(self, row_ind, col_ind, inv_perm_A, inv_perm_B):
        num_cells = len(row_ind)
        predicted_matching = np.zeros((num_cells, num_cells), dtype=np.float32)
        predicted_matching[row_ind, col_ind] = 1.0
        inv_perm_A = inv_perm_A.cpu().numpy()
        inv_perm_B = inv_perm_B.cpu().numpy()
        ideal_matching = np.zeros((num_cells, num_cells), dtype=np.float32)
        ideal_matching[inv_perm_A, inv_perm_B] = 1.0
        correct_matches = int((predicted_matching * ideal_matching).sum())
        accuracy = correct_matches / num_cells if num_cells > 0 else 0
        return accuracy


# ===================================================================
# 5. 主執行程式
# ===================================================================
if __name__ == "__main__":
    # ### NEW ###: 模式控制開關
    # 設定為 True 以使用動態 lambda scheduler。
    # 設定為 False 以使用固定的 lambda 值。
    USE_LAMBDA_SCHEDULER = False

    # --- 基本參數 ---
    GRAPH_DATA_FOLDER = "./Graph_Data/OnlyGeo/Structural_geo_features_R35"
    NODE_FEATURE_DIM = 8
    GNN_HIDDEN_DIM = 256
    GNN_OUTPUT_DIM = 128
    GAT_HEADS = 8
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 1000
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    DISTANCE_TYPE = "MSE"
    DROPOUT_RATE = 0.0
    PROGRESS_BAR_REFRESH_RATE = 50

    if USE_LAMBDA_SCHEDULER:
        # --- 動態 Lambda Scheduler 參數 ---
        LAMBDA_START = 50.0
        LAMBDA_END = 200.0
        LAMBDA_WARMUP_EPOCHS = 10
        # 使用 LAMBDA_START 初始化 criterion
        lambda_initial_val = LAMBDA_START
        logger_name = "Only_Geo_Lambda_Annealing_200Epoch"
        logging.info("模式：啟用 Lambda 動態調整 (Annealing)。")
    else:
        # --- 固定 Lambda 參數 ---
        LAMBDA_VAL = 50  # 3 Layers = 50, 4 Layers = 10
        # 使用固定的 LAMBDA_VAL 初始化 criterion
        lambda_initial_val = LAMBDA_VAL
        logger_name = f"geo_features_R35_100EpochES_Lambda{LAMBDA_VAL:.2f}_LR{LEARNING_RATE:.4f}"
        logging.info(f"模式：使用固定 Lambda 值 = {LAMBDA_VAL}")

    logger = TensorBoardLogger("logs", name=logger_name)
    pl.seed_everything(42)

    # 1. 設置數據模組
    data_module = GraphPairDataModule(graph_folder=GRAPH_DATA_FOLDER, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 2. 初始化 Loss
    criterion = DifferentiableHungarianLoss(distance_type=DISTANCE_TYPE, lambda_val=lambda_initial_val)

    # 3. 實例化模型，並傳入對應的參數
    if USE_LAMBDA_SCHEDULER:
        model = GraphMatcherLightning(
            in_channels=NODE_FEATURE_DIM, hidden_channels=GNN_HIDDEN_DIM, out_channels=GNN_OUTPUT_DIM,
            learning_rate=LEARNING_RATE, criterion=criterion, heads=GAT_HEADS, dropout_rate=DROPOUT_RATE,
            # 傳入 scheduler 參數
            use_lambda_scheduler=True,
            lambda_start=LAMBDA_START,
            lambda_end=LAMBDA_END,
            lambda_warmup_epochs=LAMBDA_WARMUP_EPOCHS,
            total_epochs=NUM_EPOCHS
        )
    else:
        model = GraphMatcherLightning(
            in_channels=NODE_FEATURE_DIM, hidden_channels=GNN_HIDDEN_DIM, out_channels=GNN_OUTPUT_DIM,
            learning_rate=LEARNING_RATE, criterion=criterion, heads=GAT_HEADS, dropout_rate=DROPOUT_RATE,
            # 關閉 scheduler
            use_lambda_scheduler=False
        )

    # 4. 配置 PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=f'CheckPoints/Only_Geo/checkpoints_{logger_name}/',
                                          filename='graph-matcher-{epoch:02d}-{val_acc:.4f}',
                                          save_top_k=3,
                                          mode='min')

    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # 或 "val_acc"
        min_delta=0.00,
        patience=100,
        verbose=True,
        mode="min"  # "min" 代表指標越小越好，"max" 代表越大越好
    )

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS, accelerator="auto", num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[TQDMProgressBar(refresh_rate=PROGRESS_BAR_REFRESH_RATE),
                   checkpoint_callback,
                   early_stop_callback
        ],
        logger=logger
    )

    logging.info("\n--- 使用 PyTorch Lightning 開始訓練 ---")
    trainer.fit(model, datamodule=data_module)
    logging.info("--- 訓練完成 ---")


