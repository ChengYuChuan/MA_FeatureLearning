# validate_graphs.py

from pathlib import Path
import torch
from tqdm import tqdm


def validate_all_graphs(graph_folder: str):
    """
    遍歷所有 .pt 檔案，檢查其有效性。
    """
    graph_folder = Path(graph_folder)
    graph_files = sorted([f for f in graph_folder.glob("*.pt")])

    if not graph_files:
        print(f"在 {graph_folder} 中找不到任何 .pt 檔案。")
        return

    print(f"開始驗證 {len(graph_files)} 個圖檔案...")

    invalid_files = []

    for file_path in tqdm(graph_files, desc="Validating"):
        try:
            graph = torch.load(file_path)

            # 檢查1：節點數是否為正
            if graph.num_nodes <= 0:
                print(f"錯誤: {file_path.name} 沒有節點 (num_nodes={graph.num_nodes})")
                invalid_files.append(file_path.name)
                continue

            # 檢查2：邊是否存在
            if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                print(f"錯誤: {file_path.name} 缺少 edge_index")
                invalid_files.append(file_path.name)
                continue

            # 檢查3：邊索引是否越界 (這是最關鍵的檢查！)
            max_edge_index = graph.edge_index.max().item()
            if max_edge_index >= graph.num_nodes:
                print(f"嚴重錯誤: {file_path.name} 的 edge_index 越界！")
                print(f"  - 節點數 (num_nodes): {graph.num_nodes} (有效索引 0 到 {graph.num_nodes - 1})")
                print(f"  - 邊索引最大值 (max_edge_index): {max_edge_index}")
                invalid_files.append(file_path.name)

        except Exception as e:
            print(f"錯誤: 無法載入或處理 {file_path.name}: {e}")
            invalid_files.append(file_path.name)

    if not invalid_files:
        print("\n✅ 所有圖檔案均有效！")
    else:
        print(f"\n❌ 發現 {len(invalid_files)} 個無效的圖檔案：")
        for fname in invalid_files:
            print(f"  - {fname}")
        print("\n請修復 Graph_building.py 並刪除這些檔案後重新生成。")


if __name__ == "__main__":
    GRAPH_FOLDER = "./Graph_data60"
    validate_all_graphs(GRAPH_FOLDER)
