# main.py (Fixed)

import os
import random
from pathlib import Path
from itertools import combinations
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


from loss import DifferentiableHungarianLoss, compute_distance_matrix


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
class EmbeddingGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, consensus=True):
        super().__init__()
        self.consensus = consensus

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.1)
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
        x = F.dropout(x, p=0.1, training=self.training)

        # --- 鄰域共識增強，移到兩層GAT之間 ---
        if self.consensus:
            N = x.shape[0]
            values = torch.ones(edge_index.shape[1], device=x.device)
            A_sparse = torch.sparse_coo_tensor(edge_index, values, (N, N))
            # 這裡的 x 還是第一層的輸出，資訊更豐富
            enhanced_x = torch.sparse.mm(A_sparse, x)
            x = x + enhanced_x  # 直接作為一個殘差項加入，而不是加權平均

        # 第二層 GAT
        x = self.conv2(x, edge_index)
        x = self.norm2(x + x_shortcut)  # 外層的殘差連接
        x = F.elu(x)

        return x


# ===================================================================
# 3. 資料模組 (LightningDataModule) - 修改 DataLoader
# ===================================================================
class DynamicGraphPairDataset(Dataset):
    """
    這個 Dataset 在每次被請求一個項目時，動態地創建一個圖配對。
    """

    def __init__(self, graph_files):
        self.graph_files = graph_files  # 儲存的是單個圖的檔案列表
        self.num_graphs = len(self.graph_files)

    def __len__(self):
        # Dataset 的長度現在是圖的數量，這定義了一個 Epoch 的長度
        return self.num_graphs

    def __getitem__(self, idx):
        # 1. 取得錨點圖 (anchor graph)
        path1 = self.graph_files[idx]

        # 2. 隨機選擇另一個圖作為配對
        #    確保不會選到自己
        rand_idx = random.randint(0, self.num_graphs - 1)
        while rand_idx == idx:
            rand_idx = random.randint(0, self.num_graphs - 1)
        path2 = self.graph_files[rand_idx]

        # 3. 載入並返回配對
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

        # === 關鍵修改：直接對圖檔案列表進行切分 ===
        rng = random.Random(self.seed)
        rng.shuffle(graph_files)

        split_idx = int(len(graph_files) * (1 - self.val_split))
        self.train_files = graph_files[:split_idx]
        self.val_files = graph_files[split_idx:]
        # ===============================================

        print(f"資料集切分完成：訓練圖 {len(self.train_files)} 個，驗證圖 {len(self.val_files)} 個。")

    def train_dataloader(self):
        # 使用新的動態配對 Dataset
        dataset = DynamicGraphPairDataset(self.train_files)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 打亂錨點圖的順序
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=graph_pair_collate
        )

    def val_dataloader(self):
        # 驗證集也使用動態配對
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
# 4. 核心訓練模組 (LightningModule) - 修改 _common_step
# ===================================================================
class GraphMatcherLightning(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate, criterion, heads=4):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])
        # 啟用鄰域共識
        self.gnn_encoder = EmbeddingGAT(in_channels, hidden_channels, out_channels, heads, consensus=False)
        self.criterion = criterion

    def forward(self, graph_data):
        return self.gnn_encoder(graph_data)

    def _common_step(self, batch, batch_idx):
        # --- MODIFICATION: 大幅簡化 ---
        graph1, graph2 = batch

        # GNN Encoder 現在內部處理了所有特徵提取和增強
        feats1 = self(graph1)
        feats2 = self(graph2)

        # 確保節點數相同
        if feats1.shape[0] != feats2.shape[0]:
            # 在實際應用中，您可能需要對齊或填充節點
            print(f"警告：節點數不匹配，跳過此批次。Graph1: {feats1.shape[0]}, Graph2: {feats2.shape[0]}")
            return None, None

        latent_features = torch.stack([feats1, feats2], dim=0)
        # --- END MODIFICATION ---

        inv_perm1 = torch.argsort(graph1.perm).to(self.device)
        inv_perm2 = torch.argsort(graph2.perm).to(self.device)

        loss, (row_ind, col_ind) = self.criterion(latent_features, inv_perm_A=inv_perm1, inv_perm_B=inv_perm2)
        acc = self._calculate_accuracy(row_ind, col_ind, inv_perm1, inv_perm2)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        if loss is not None:
            # === 關鍵修改：明確指定 batch_size ===
            current_batch_size = 1 # 因為我們每次處理一個配對
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            # ========================================
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        if loss is not None:
            # === 關鍵修改：明確指定 batch_size ===
            current_batch_size = 1 # 因為我們每次處理一個配對
            self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=current_batch_size)
            self.log('val_acc', acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=current_batch_size)
            # ========================================
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',  # 監控的指標越小越好
        #     factor=0.5,  # 每次降低學習率為原來的一半
        #     patience=5,  # 5個epoch驗證損失沒有改善就觸發
        #     verbose=True
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",  # 監控的指標
        #     },
        # }

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
    logger = TensorBoardLogger("logs", name="graph_matching")

    GRAPH_FOLDER = "./Graph_data_standardized_structural_05"
    NODE_FEATURE_DIM = 3
    GNN_HIDDEN_DIM = 256
    GNN_OUTPUT_DIM = 128
    GAT_HEADS = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    LAMBDA_VAL = 1
    DISTANCE_TYPE = "MSE"

    VAL_CHECK_INTERVAL = 1.0
    LOG_EVERY_N_STEPS = 10
    PROGRESS_BAR_REFRESH_RATE = 100

    pl.seed_everything(42)

    data_module = GraphPairDataModule(graph_folder=GRAPH_FOLDER, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    criterion = DifferentiableHungarianLoss(distance_type=DISTANCE_TYPE, lambda_val=LAMBDA_VAL)

    model = GraphMatcherLightning(
        in_channels=NODE_FEATURE_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUTPUT_DIM,
        learning_rate=LEARNING_RATE,
        criterion=criterion,
        heads=GAT_HEADS
    )
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath='checkpoints/',
                                          filename='graph-matcher-{epoch:02d}-{val_acc:.2f}',
                                          save_top_k=3,
                                          mode='min')
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        num_sanity_val_steps=0,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        callbacks=[TQDMProgressBar(refresh_rate=PROGRESS_BAR_REFRESH_RATE),
                   checkpoint_callback],
        logger = logger
    )

    # 為了更好的除錯，設定環境變量
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    print("\n--- 使用 PyTorch Lightning 和端到端損失函數開始訓練 ---")
    trainer.fit(model, datamodule=data_module)
    print("--- 訓練完成 ---")

