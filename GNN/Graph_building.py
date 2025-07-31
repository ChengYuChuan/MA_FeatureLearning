import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.preprocessing import StandardScaler
import networkx as nx  # <--- 引入 networkx 來計算圖特徵

import torch
from torch_geometric.data import Data

# ====================
# Logging 設定
# ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("graph_building_structural_features.log"),
        logging.StreamHandler()
    ]
)


# 建立 PyG Graph 的函式
def build_graph(df, cell_features=None, method='radius', k=10, radius=0.5, min_degree=3):
    """
    建立 PyG Graph，對座標進行Z-score標準化，並使用結構特徵作為節點輸入。
    同時提供了串接細胞特徵的擴充點。
    """
    coords_original = df[['x', 'y', 'z']].values
    scaler_pos = StandardScaler()
    coords_scaled = scaler_pos.fit_transform(coords_original)

    aligned_idx = df['Aligned_No'].values - 1
    N = len(coords_original)
    label_names = df['label_name'].tolist()
    y_raw = np.arange(N)

    # 1. 建立圖的邊 (Edges) - 在這個階段，我們使用 0 到 N-1 的原始索引
    # 這樣有利於後續使用 networkx 計算特徵

    # 注意：這裡的 knn 和 radius_neighbors_graph 是對標準化後的座標進行操作
    # 但返回的索引是相對於輸入陣列的，也就是 0 到 N-1
    edges = set()
    if method == 'radius':
        A = radius_neighbors_graph(coords_scaled, radius=radius, include_self=False)
        coo = A.tocoo()
        for a, b in zip(coo.row, coo.col):
            edges.add(tuple(sorted((a, b))))
    elif method == 'knn':
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords_scaled)
        _, indices = nbrs.kneighbors(coords_scaled)
        for i in range(N):
            for j in indices[i][1:]:
                edges.add(tuple(sorted((i, j))))

    # 補全低度節點
    degree_count = {i: 0 for i in range(N)}
    for a, b in edges:
        degree_count[a] += 1
        degree_count[b] += 1

    low_degree_nodes = [i for i, d in degree_count.items() if d < min_degree]
    if low_degree_nodes:
        nbrs = NearestNeighbors(n_neighbors=min_degree + 1).fit(coords_scaled)
        # 只對低度節點查詢鄰居
        _, indices = nbrs.kneighbors(coords_scaled[low_degree_nodes])
        for i, node_idx in enumerate(low_degree_nodes):
            for neighbor_idx in indices[i][1:]:
                edges.add(tuple(sorted((node_idx, neighbor_idx))))

    # 2. 計算結構特徵 (基於已經建立好的邊)
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(N))
    G_nx.add_edges_from(list(edges))

    node_degrees = np.array([G_nx.degree(i) for i in range(N)]).reshape(-1, 1)
    clustering_coeffs = np.array(list(nx.clustering(G_nx).values())).reshape(-1, 1)
    pageranks = np.array(list(nx.pagerank(G_nx).values())).reshape(-1, 1)

    # 組合結構特徵並進行標準化，這是個好習慣
    structural_features = np.concatenate([node_degrees, clustering_coeffs, pageranks], axis=1).astype(np.float32)
    scaler_struct_feat = StandardScaler()
    structural_features_scaled = scaler_struct_feat.fit_transform(structural_features)

    # 3. 準備最終的節點特徵 x
    # =========================================================================
    # === 未來串接細胞特徵的擴充點 ===
    # =========================================================================
    if cell_features is not None:
        # 假設 cell_features 是一個 [N, F_cell] 的 NumPy 陣列

        # a. 對細胞特徵也進行獨立的標準化
        scaler_cell_feat = StandardScaler()
        cell_features_scaled = scaler_cell_feat.fit_transform(cell_features)

        # b. 將結構特徵和細胞特徵串接起來
        print("INFO: 串接結構特徵與細胞特徵。")
        final_features = np.concatenate([structural_features_scaled, cell_features_scaled], axis=1)
    else:
        # 如果沒有細胞特徵，就只使用結構特徵
        print("INFO: 僅使用結構特徵。")
        final_features = structural_features_scaled
    # =========================================================================

    x_raw = final_features

    # 4. 隨機打亂節點順序 (Permutation)
    # 這是最後一步，確保所有數據（x, pos, y 等）都以相同的順序被打亂
    perm = np.random.permutation(N)

    if len(label_names) != N:
        raise ValueError("label_names length and numbers of nodes are not the same")

    x = torch.tensor(x_raw[perm], dtype=torch.float)
    pos = torch.tensor(coords_scaled[perm], dtype=torch.float)
    original_pos = torch.tensor(coords_original[perm], dtype=torch.float)
    y = torch.tensor(y_raw[perm], dtype=torch.long)
    node_names = [label_names[i] for i in perm]

    # 建立一個從原始索引到 perm 索引的映射
    perm_map = {original_idx: perm_idx for perm_idx, original_idx in enumerate(perm)}

    # 使用映射轉換 edge_index
    edge_list_permuted = [(perm_map[u], perm_map[v]) for u, v in edges]
    edge_index = torch.tensor(edge_list_permuted, dtype=torch.long).t().contiguous()

    return Data(
        x=x,
        pos=pos,
        original_pos=original_pos,
        edge_index=edge_index,
        y=y,
        node_names=node_names,
        perm=torch.tensor(perm, dtype=torch.long)
    )


# 主批次處理函式
def process_all_graphs(df_folder, feature_folder, saving_folder, method='radius', k=10, radius=0.5):
    processed = 0
    failed = []

    df_folder = Path(df_folder)
    saving_folder = Path(saving_folder)
    saving_folder.mkdir(parents=True, exist_ok=True)

    if feature_folder is not None:
        feature_folder = Path(feature_folder)

    for file in tqdm(os.listdir(df_folder)):
        if file.endswith(".txt"):
            worm_id = file.replace(".txt", "")
            try:
                txt_path = df_folder / file
                save_path = saving_folder / f"{worm_id}.pt"

                # === 讀取細胞特徵的入口 ===
                cell_feat = None
                if feature_folder is not None:
                    npz_path = feature_folder / f"{worm_id}.npz"
                    if npz_path.exists():
                        npz_data = np.load(npz_path)
                        # 假設特徵存在 "features" 這個 key 下
                        cell_feat = npz_data.get("features")

                if not txt_path.exists():
                    logging.warning(f"[跳過] 缺少 TXT：{txt_path}")
                    continue

                df = pd.read_csv(txt_path, sep="\t")

                # 將細胞特徵傳入 build_graph 函式
                graph = build_graph(df, cell_features=cell_feat, method=method, k=k, radius=radius)
                graph.worm_id = worm_id
                torch.save(graph, save_path)

                logging.info(f"[完成] {worm_id} → {save_path}")
                processed += 1

            except Exception as e:
                logging.error(f"[錯誤] {worm_id} 發生錯誤：{e}", exc_info=True)
                failed.append(worm_id)

    logging.info(f"\n✅ 共處理 {processed} 隻蟲的圖")
    if failed:
        logging.warning(f"⚠️ 有 {len(failed)} 筆失敗：{failed}")


# 若直接執行此腳本
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    logging.info(f"Base Dir: {base_dir}")

    df_folder = base_dir / "Dict_of_Cells"
    # === 如果您有細胞特徵，請指定資料夾路徑 ===
    # features_folder = base_dir / "Cell_Features"
    features_folder = None  # 目前先設為 None

    saving_folder = base_dir / "Graph_data_standardized_structural_05"

    process_all_graphs(df_folder, features_folder, saving_folder, method='radius', radius=0.5)

