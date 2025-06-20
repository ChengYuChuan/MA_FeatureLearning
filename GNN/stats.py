import torch
from pathlib import Path
from collections import defaultdict

graph_path = Path("Graph_data60")
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

# only checking three sample
# import torch
# import random
# from pathlib import Path
# from collections import defaultdict
#
# graph_path = Path("Graph_data60")
# pt_files = list(graph_path.glob("*.pt"))
#
# # 隨機抽取 3 張圖（如果圖檔小於3也會保證不出錯）
# sample_files = random.sample(pt_files, k=min(3, len(pt_files)))
#
# # 初始化統計量
# all_degrees = []
# max_degree_names = []
# min_degree_names = []
#
# for pt_file in sample_files:
#     graph = torch.load(pt_file, map_location='cpu')
#     edge_index = graph.edge_index
#     node_names = graph.node_names
#
#     # 建立無向邊集合
#     edges = set()
#     for i in range(edge_index.size(1)):
#         a, b = sorted((edge_index[0, i].item(), edge_index[1, i].item()))
#         edges.add((a, b))
#
#     # 統計每個節點的 degree
#     degree_count = defaultdict(int)
#     for a, b in edges:
#         degree_count[a] += 1
#         degree_count[b] += 1
#
#     degrees = list(degree_count.values())
#     all_degrees.extend(degrees)
#
#     max_deg = max(degrees)
#     min_deg = min(degrees)
#
#     max_degree_names.extend([node_names[i] for i in degree_count if degree_count[i] == max_deg])
#     min_degree_names.extend([node_names[i] for i in degree_count if degree_count[i] == min_deg])
#
# # 統整報告
# print("🎯 統計基於以下圖檔:")
# for f in sample_files:
#     print(" -", f.name)
#
# print(f"\n📈 最大邊數: {max(all_degrees)}")
# print(f"📉 最小邊數: {min(all_degrees)}")
# print(f"📊 平均邊數: {sum(all_degrees) / len(all_degrees):.2f}")
# print(f"🧬 最多邊數的細胞: {set(max_degree_names)}")
# print(f"🪶 最少邊數的細胞: {set(min_degree_names)}")
