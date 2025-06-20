# main.py (Fixed)

import os
import random
from pathlib import Path
from itertools import combinations
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from loss import DifferentiableHungarianLoss


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
# 2. 核心模型定義 (GNN Encoder) - 無需修改
# ===================================================================
class EmbeddingGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph_data: Data):
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# ===================================================================
# 3. 資料模組 (LightningDataModule) - 修改 DataLoader
# ===================================================================
class GraphPairDataset(Dataset):
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
    def __init__(self, graph_folder: str, batch_size: int = 1, val_split: float = 0.2, seed: int = 42,
                 num_workers: int = 0):
        super().__init__()
        self.graph_folder = Path(graph_folder)
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.num_workers = num_workers
        self.train_pairs = []
        self.val_pairs = []

    def setup(self, stage: str = None):
        graph_files = sorted([f for f in self.graph_folder.glob("*.pt")])
        if not graph_files:
            raise FileNotFoundError(f"在 {self.graph_folder} 中找不到任何 .pt 圖檔案。")

        all_pairs = list(combinations(graph_files, 2))

        rng = random.Random(self.seed)
        rng.shuffle(all_pairs)

        split_idx = int(len(all_pairs) * (1 - self.val_split))
        self.train_pairs = all_pairs[:split_idx]
        self.val_pairs = all_pairs[split_idx:]

        print(f"資料集切分完成：訓練集 {len(self.train_pairs)} 對，驗證集 {len(self.val_pairs)} 對。")

    def train_dataloader(self):
        dataset = GraphPairDataset(self.train_pairs)
        # <--- MODIFICATION: 加入 collate_fn
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=graph_pair_collate
        )

    def val_dataloader(self):
        dataset = GraphPairDataset(self.val_pairs)
        # <--- MODIFICATION: 加入 collate_fn
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=graph_pair_collate
        )


# ===================================================================
# 4. 核心訓練模組 (LightningModule) - 修改 _common_step
# ===================================================================
class GraphMatcherLightning(pl.LightningModule):
    def __init__(self, node_feature_dim: int, gnn_hidden_dim: int, gnn_output_dim: int, learning_rate: float,
                 criterion: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])
        self.gnn_encoder = EmbeddingGNN(node_feature_dim, gnn_hidden_dim, gnn_output_dim)
        self.criterion = criterion

    def forward(self, graph_data):
        return self.gnn_encoder(graph_data)

    def _common_step(self, batch, batch_idx):
        # <--- MODIFICATION: 由於使用了自訂 collate_fn，batch 現在直接是 (graph1, graph2) 元組
        graph1, graph2 = batch

        feats1 = self(graph1)
        feats2 = self(graph2)

        assert feats1.shape[0] == feats2.shape[0], "此實現假設配對的圖節點數相同"
        latent_features = torch.stack([feats1, feats2], dim=0)

        inv_perm1 = torch.argsort(graph1.perm).to(self.device)
        inv_perm2 = torch.argsort(graph2.perm).to(self.device)

        loss, (row_ind, col_ind) = self.criterion(latent_features, inv_perm_A=inv_perm1, inv_perm_B=inv_perm2)
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
# 5. 主執行程式 - 無需修改
# ===================================================================
if __name__ == "__main__":
    GRAPH_FOLDER = "./Graph_data60"
    NODE_FEATURE_DIM = 512
    GNN_HIDDEN_DIM = 256
    GNN_OUTPUT_DIM = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    LAMBDA_VAL = 20
    DISTANCE_TYPE = "MSE"

    VAL_CHECK_INTERVAL = 1
    LOG_EVERY_N_STEPS = 10
    PROGRESS_BAR_REFRESH_RATE = 10

    pl.seed_everything(42)

    data_module = GraphPairDataModule(
        graph_folder=GRAPH_FOLDER,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    criterion = DifferentiableHungarianLoss(
        distance_type=DISTANCE_TYPE,
        lambda_val=LAMBDA_VAL
    )

    model = GraphMatcherLightning(
        node_feature_dim=NODE_FEATURE_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_output_dim=GNN_OUTPUT_DIM,
        learning_rate=LEARNING_RATE,
        criterion=criterion
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='graph-matcher-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        val_check_interval=VAL_CHECK_INTERVAL,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        callbacks=[TQDMProgressBar(refresh_rate=PROGRESS_BAR_REFRESH_RATE),
                   checkpoint_callback]
    )

    print("\n--- 使用 PyTorch Lightning 和端到端損失函數開始訓練 ---")
    trainer.fit(model, datamodule=data_module)
    print("--- 訓練完成 ---")

