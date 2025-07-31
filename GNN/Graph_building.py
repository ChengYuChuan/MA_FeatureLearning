import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.preprocessing import StandardScaler
import networkx as nx 

from collections import defaultdict

import torch
from torch_geometric.data import Data

# ====================
# Logging 設定
# ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("graph_building.log"),  # ### MODIFIED ### 日誌文件名簡化
        logging.StreamHandler()
    ]
)


# 建立 PyG Graph 的函式
def build_graph(df, cell_features=None, method='radius', k=10, radius=60, min_degree=3):
    coords_original = df[['x', 'y', 'z']].values
    N = len(coords_original)

    # 計算相對位置特徵
    x_coords = coords_original[:, 0]
    y_coords = coords_original[:, 1]
    z_coords = coords_original[:, 2]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    epsilon = 1e-9
    x_relative_pos = (x_coords - x_min) / (x_max - x_min + epsilon) if (x_max - x_min) > 0 else np.zeros_like(x_coords)
    y_relative_pos = (y_coords - y_min) / (y_max - y_min + epsilon) if (y_max - y_min) > 0 else np.zeros_like(y_coords)
    z_relative_pos = (z_coords - z_min) / (z_max - z_min + epsilon) if (z_max - z_min) > 0 else np.zeros_like(z_coords)

    # 建立圖（全部用原始物理座標）
    edges = set()
    if method == 'radius':
        A = radius_neighbors_graph(coords_original, radius=radius, include_self=False)
        coo = A.tocoo()
        for a, b in zip(coo.row, coo.col):
            edges.add(tuple(sorted((a, b))))
    elif method == 'knn':
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords_original)
        _, indices = nbrs.kneighbors(coords_original)
        for i in range(N):
            for j in indices[i][1:]:
                edges.add(tuple(sorted((i, j))))
    else:
        raise ValueError("method must be 'knn' or 'radius'")

    # 補全低度節點
    degree_count = {i: 0 for i in range(N)}
    for a, b in edges:
        degree_count[a] += 1
        degree_count[b] += 1
    low_degree_nodes = [i for i, d in degree_count.items() if d < min_degree]
    if low_degree_nodes:
        nbrs = NearestNeighbors(n_neighbors=min_degree + 1).fit(coords_original)
        _, indices = nbrs.kneighbors(coords_original[low_degree_nodes])
        for i, node_idx in enumerate(low_degree_nodes):
            for neighbor_idx in indices[i][1:]:
                edges.add(tuple(sorted((node_idx, neighbor_idx))))

    # 結構特徵
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(N))
    G_nx.add_edges_from(list(edges))
    node_degrees = np.array([G_nx.degree(i) for i in range(N)]).reshape(-1, 1)
    clustering_coeffs = np.array(list(nx.clustering(G_nx).values())).reshape(-1, 1)
    pageranks = np.array(list(nx.pagerank(G_nx).values())).reshape(-1, 1)
    structural_features = np.concatenate([node_degrees, clustering_coeffs, pageranks], axis=1).astype(np.float32)
    scaler_struct_feat = StandardScaler()
    structural_features_scaled = scaler_struct_feat.fit_transform(structural_features)

    # 幾何特徵
    centroid = coords_original.mean(axis=0)
    dist_to_centroid_full_space = np.linalg.norm(coords_original - centroid, axis=1).reshape(-1, 1)
    x_pos_forward = x_relative_pos.reshape(-1, 1)
    x_pos_backward = (1 - x_relative_pos).reshape(-1, 1)
    y_pos_relative = y_relative_pos.reshape(-1, 1)
    z_pos_relative = z_relative_pos.reshape(-1, 1)
    geometric_features = np.concatenate([
        dist_to_centroid_full_space,
        x_pos_forward,
        x_pos_backward,
        y_pos_relative,
        z_pos_relative
    ], axis=1).astype(np.float32)
    scaler_geo_feat = StandardScaler()
    geometric_features_scaled = scaler_geo_feat.fit_transform(geometric_features)

    # 結構特徵 + 幾何特徵
    base_features = np.concatenate([structural_features_scaled, geometric_features_scaled], axis=1)

    # 加入細胞特徵
    if cell_features is not None:
        scaler_cell_feat = StandardScaler()
        if cell_features.ndim == 1:
            cell_features = cell_features.reshape(-1, 1)
        cell_features_scaled = scaler_cell_feat.fit_transform(cell_features)
        final_features = np.concatenate([base_features, cell_features_scaled], axis=1)
    else:
        final_features = base_features

    x_raw = final_features

    # 隨機打亂節點順序
    perm = np.random.permutation(N)
    label_names = df['label_name'].tolist()
    y_raw = np.arange(N)
    x = torch.tensor(x_raw[perm], dtype=torch.float)
    pos = torch.tensor(coords_original[perm], dtype=torch.float)  # 這裡用原始座標
    y = torch.tensor(y_raw[perm], dtype=torch.long)
    node_names = [label_names[i] for i in perm]
    perm_map = {original_idx: perm_idx for perm_idx, original_idx in enumerate(perm)}
    edge_list_permuted = [(perm_map[u], perm_map[v]) for u, v in edges]
    edge_index = torch.tensor(list(edge_list_permuted), dtype=torch.long).t().contiguous()

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        y=y,
        node_names=node_names,
        perm=torch.tensor(perm, dtype=torch.long)
    )

# 將 process_all_graphs 移到 build_graph.py 檔案，方便呼叫
def process_all_graphs(df_folder, feature_folder, saving_folder, method='radius', k=10, radius=45):
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

                cell_feat = None
                if feature_folder is not None:
                    npz_path = feature_folder / f"{worm_id}.npz"
                    if npz_path.exists():
                        npz_data = np.load(npz_path)
                        cell_feat = npz_data.get("features")

                if not txt_path.exists():
                    logging.warning(f"[跳過] 缺少 TXT：{txt_path}")
                    continue

                df = pd.read_csv(txt_path, sep="\t")

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




if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    logging.info(f"Base Dir: {base_dir}")

    df_folder = base_dir / "Dict_of_Cells"
    # features_folder = base_dir / "ExtractedFeatures" / "DoubleConv" /"MLP512 8 3Layers MSE"
    features_folder = None
    saving_folder = base_dir / "Graph_Data" / "Graph_data_structural_geo_features_R55_noCellFeature"

    # 執行數據生成
    RADIUS = 55
    print(f"Radius we are using : {RADIUS}")
    process_all_graphs(df_folder, features_folder, saving_folder, method='radius', radius=RADIUS)

    graph_path = saving_folder
    pt_files = list(graph_path.glob("*.pt"))

    # 初始化統計量
    all_degrees = []
    max_degree_names = []
    min_degree_names = []

    for pt_file in pt_files:
        graph = torch.load(pt_file, map_location='cpu')
        edge_index = graph.edge_index
        node_names = graph.node_names

        # 建立無向邊集合
        edges = set()
        for i in range(edge_index.size(1)):
            a, b = sorted((edge_index[0, i].item(), edge_index[1, i].item()))
            edges.add((a, b))

        # 統計每個節點的 degree
        degree_count = defaultdict(int)
        for a, b in edges:
            degree_count[a] += 1
            degree_count[b] += 1

        degrees = list(degree_count.values())
        all_degrees.extend(degrees)

        max_deg = max(degrees)
        min_deg = min(degrees)

        max_degree_names.extend([node_names[i] for i in degree_count if degree_count[i] == max_deg])
        min_degree_names.extend([node_names[i] for i in degree_count if degree_count[i] == min_deg])

    # 統整報告
    print(f"📈 最大邊數: {max(all_degrees)}")
    print(f"📉 最小邊數: {min(all_degrees)}")
    print(f"📊 平均邊數: {sum(all_degrees) / len(all_degrees):.2f}")
    print(f"🧬 最多邊數的細胞: {set(max_degree_names)}")
    print(f"🪶 最少邊數的細胞: {set(min_degree_names)}")
