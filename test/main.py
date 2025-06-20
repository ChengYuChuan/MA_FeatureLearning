# main.py

import os
import random
from pathlib import Path
from itertools import combinations
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data  # 導入 PyG 的 Data 類

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# ===================================================================
# 引用您提供的 loss.py 中的核心類別
# 請確保 loss.py 與這個 main.py 在同一個資料夾下
# ===================================================================
from loss import DifferentiableHungarianLoss, LAPSolver, HammingLoss, compute_distance_matrix


# ===================================================================
# 1. 核心模型定義 (GNN Encoder)
# ===================================================================

class EmbeddingGNN(nn.Module):
    """
    一個簡單的GNN編碼器，將圖的節點特徵轉換為嵌入向量。
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph_data: Data):
        # 從 PyG 的 Data 物件中提取特徵和邊
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# ===================================================================
# 2. 資料模組 (LightningDataModule) - 實現顯式配對
# ===================================================================

class GraphPairDataset(Dataset):
    """
    這個 Dataset 類別現在只專注於根據給定的圖檔案配對來提供資料。
    """

    def __init__(self, graph_pairs):
        self.graph_pairs = graph_pairs

    def __len__(self):
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        path1, path2 = self.graph_pairs[idx]
        graph1 = torch.load(path1)
        graph2 = torch.load(path2)
        return graph1, graph2


class GraphPairDataModule(pl.LightningDataModule):
    def __init__(self, graph_folder: str, batch_size: int = 1, val_split: float = 0.2, seed: int = 42):
        super().__init__()
        self.graph_folder = Path(graph_folder)
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.train_pairs = []
        self.val_pairs = []

    def setup(self, stage: str = None):
        graph_files = sorted([f for f in self.graph_folder.glob("*.pt")])
        if not graph_files:
            raise FileNotFoundError(f"在 {self.graph_folder} 中找不到任何 .pt 圖檔案。")

        all_pairs = list(combinations(graph_files, 2))

        # 使用固定的種子來確保每次執行的切分都一樣
        rng = random.Random(self.seed)
        rng.shuffle(all_pairs)

        split_idx = int(len(all_pairs) * (1 - self.val_split))
        self.train_pairs = all_pairs[:split_idx]
        self.val_pairs = all_pairs[split_idx:]

        print(f"資料集切分完成：訓練集 {len(self.train_pairs)} 對，驗證集 {len(self.val_pairs)} 對。")

    def train_dataloader(self):
        dataset = GraphPairDataset(self.train_pairs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = GraphPairDataset(self.val_pairs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


# ===================================================================
# 3. 核心訓練模組 (LightningModule) - 整合GNN與您的損失函數
# ===================================================================

class GraphMatcherLightning(pl.LightningModule):
    def __init__(self, node_feature_dim: int, gnn_hidden_dim: int, gnn_output_dim: int, learning_rate: float,
                 criterion: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])
        self.gnn_encoder = EmbeddingGNN(node_feature_dim, gnn_hidden_dim, gnn_output_dim)
        self.criterion = criterion

    def forward(self, graph_data):
        # 這個 forward 方法現在處理單個圖
        return self.gnn_encoder(graph_data)

    def _common_step(self, batch, batch_idx):
        # 由於 batch_size=1，batch 是一個包含 (graph1, graph2) 的列表
        graph1, graph2 = batch[0]

        # 1. 分別通過 GNN Encoder 得到兩組節點嵌入
        feats1 = self(graph1)  # Shape: [N, gnn_output_dim]
        feats2 = self(graph2)  # Shape: [M, gnn_output_dim]

        # 2. 將兩組嵌入合併成損失函數需要的格式 (2, N, F)
        # 假設兩張圖的節點數相同 (N=M)
        assert feats1.shape[0] == feats2.shape[0], "此實現假設配對的圖節點數相同"
        latent_features = torch.stack([feats1, feats2], dim=0)

        # 3. 準備 inv_perms
        # 我們的目標是將打亂的節點順序恢復到它們的生物學順序 (Aligned_No)
        # `graph.y` 儲存了原始的生物學索引 (0 to N-1)
        # `graph.perm` 儲存了打亂後的順序
        # 我們需要的是 `inv_perm`，可以將打亂後的索引映射回原始索引

        # 計算 inv_perm: inv_perm[perm[i]] = i
        inv_perm1 = torch.argsort(graph1.perm).to(self.device)
        inv_perm2 = torch.argsort(graph2.perm).to(self.device)

        # 4. 計算損失
        loss, (row_ind, col_ind) = self.criterion(latent_features, inv_perm_A=inv_perm1, inv_perm_B=inv_perm2)

        # 5. 計算準確率
        acc = self._calculate_accuracy(row_ind, col_ind, inv_perm1, inv_perm2)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def _calculate_accuracy(self, row_ind, col_ind, inv_perm_A, inv_perm_B):
        # 這個方法直接從您的 lightningLAPNetwMLP.py 中借用，無需修改
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
# 4. 主執行程式 - 使用 Trainer 啟動訓練
# ===================================================================
if __name__ == "__main__":
    # --- 超參數設定 ---
    GRAPH_FOLDER = "./Graph_data60"
    NODE_FEATURE_DIM = 512
    GNN_HIDDEN_DIM = 256
    GNN_OUTPUT_DIM = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    BATCH_SIZE = 1  # 每個 batch 包含一對圖
    LAMBDA_VAL = 20  # DifferentiableHungarianLoss 的超參數
    DISTANCE_TYPE = "MSE"  # 成本矩陣的計算方式

    pl.seed_everything(42)  # 為了可重複性

    # 1. 實例化資料模組
    data_module = GraphPairDataModule(graph_folder=GRAPH_FOLDER, batch_size=BATCH_SIZE)

    # 2. 實例化損失函數
    criterion = DifferentiableHungarianLoss(
        distance_type=DISTANCE_TYPE,
        lambda_val=LAMBDA_VAL
    )

    # 3. 實例化模型模組
    model = GraphMatcherLightning(
        node_feature_dim=NODE_FEATURE_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_output_dim=GNN_OUTPUT_DIM,
        learning_rate=LEARNING_RATE,
        criterion=criterion
    )

    # 4. 設定模型儲存的回調函式
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='graph-matcher-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    # 5. 實例化並設定 Trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        log_every_n_steps=5,
        callbacks=[checkpoint_callback]
    )

    # 6. 啟動訓練！
    print("\n--- 使用 PyTorch Lightning 和端到端損失函數開始訓練 ---")
    trainer.fit(model, datamodule=data_module)
    print("--- 訓練完成 ---")

