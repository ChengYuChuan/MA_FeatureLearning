import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

import torch
from torch_geometric.data import Data


# ====================
# Logging 設定
# ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("graph_building.log"),
        logging.StreamHandler()
    ]
)


# 建立 PyG Graph 的函式
def build_graph(df, feature_matrix=None, method='radius', k=10, radius=30.0, min_degree=3):
    coords = df[['x', 'y', 'z']].values
    aligned_idx = df['Aligned_No'].values - 1
    N = len(coords)

    if feature_matrix is None:
        x_raw = np.ones((N, 512), dtype=np.float32)
    else:
        x_raw = feature_matrix[aligned_idx]

    pos_raw = coords
    label_names = df['label_name'].tolist()
    y_raw = np.arange(N)

    perm = np.random.permutation(N)

    if len(label_names) != N:
        raise ValueError("label_names length and numbers of nodes are not the same")

    x = torch.tensor(x_raw[perm], dtype=torch.float)
    pos = torch.tensor(pos_raw[perm], dtype=torch.float)
    y = torch.tensor(y_raw[perm], dtype=torch.long)
    node_names = [label_names[i] for i in perm]

    # 1. 建立 radius-based 邊
    edges = set()
    if method == 'radius':
        from sklearn.neighbors import radius_neighbors_graph
        A = radius_neighbors_graph(pos.numpy(), radius=radius, include_self=False)
        coo = A.tocoo()
        for a, b in zip(coo.row, coo.col):
            edges.add(tuple(sorted((a, b))))

    elif method == 'knn':
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(pos.numpy())
        _, indices = nbrs.kneighbors(pos.numpy())
        for i in range(N):
            for j in indices[i][1:]:
                edges.add(tuple(sorted((i, j))))
    else:
        raise ValueError("method must be 'knn' or 'radius'")

    # 2. 補 degree < min_degree 的節點（使用 knn）
    degree_count = [0] * N
    for a, b in edges:
        degree_count[a] += 1
        degree_count[b] += 1

    low_degree_nodes = [i for i, d in enumerate(degree_count) if d < min_degree]

    if low_degree_nodes:
        nbrs = NearestNeighbors(n_neighbors=min_degree+1).fit(pos.numpy())
        _, indices = nbrs.kneighbors(pos.numpy())
        for i in low_degree_nodes:
            for j in indices[i][1:]:
                edges.add(tuple(sorted((i, j))))

    edge_index = torch.tensor(sorted(edges)).t().contiguous()

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        y=y,
        node_names=node_names,
        perm=torch.tensor(perm, dtype=torch.long)
    )




# 主批次處理函式
def process_all_graphs(df_folder, feature_folder, saving_folder, method='knn', k=10, radius=30.0):
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
                # npz_path = feature_folder / f"{worm_id}.npz"
                save_path = saving_folder / f"{worm_id}.pt"

                # 暫時不用特徵
                features = None

                if not txt_path.exists():
                    logging.warning(f"[跳過] 缺少 TXT：{txt_path}")
                    continue

                df = pd.read_csv(txt_path, sep="\t")

                # 若要使用真實特徵，改為讀取 npz：
                # if npz_path.exists():
                #     npz_data = np.load(npz_path)
                #     features = npz_data["features"] if "features" in npz_data else npz_data[list(npz_data.files)[0]]
                # else:
                #     logging.warning(f"[跳過] 無對應特徵檔案：{npz_path}")
                #     continue

                graph = build_graph(df, features, method=method, k=k, radius=radius)
                graph.worm_id = worm_id
                torch.save(graph, save_path)

                logging.info(f"[完成] {worm_id} → {save_path}")
                processed += 1

            except Exception as e:
                logging.error(f"[錯誤] {worm_id} 發生錯誤：{e}")
                failed.append(worm_id)

    logging.info(f"\n✅ 共處理 {processed} 隻蟲的圖")
    if failed:
        logging.warning(f"⚠️ 有 {len(failed)} 筆失敗：{failed}")


# 若直接執行此腳本
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    logging.info(f"Base Dir: {base_dir}")

    # Features_folder = None
    df_folder = base_dir / "Dict_of_Cells"
    saving_folder = base_dir / "Graph_data60"

    process_all_graphs(df_folder, None, saving_folder, method='radius', radius=60.0)
