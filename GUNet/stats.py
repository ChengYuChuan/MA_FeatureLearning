import numpy as np
import os

folder_path = '/home/students/cheng/Cubes32'  # ← 修改成你的資料夾路徑
output_file = 'npy_statistics.txt'

all_data = []
global_min = float('inf')
global_max = float('-inf')

with open(output_file, 'w') as f:
    f.write("Per-file min and max:\n")
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            data = np.load(os.path.join(folder_path, file_name))
            file_min = data.min()
            file_max = data.max()
            f.write(f"{file_name}: min = {file_min}, max = {file_max}\n")

            print(f"{file_name}: min = {file_min}, max = {file_max}\n")

            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)
            all_data.append(data)

    f.write("\nGlobal statistics across all files:\n")
    all_data = np.stack(all_data)
    mean = all_data.mean()
    std = all_data.std()
    f.write(f"Overall min: {global_min}\n")
    f.write(f"Overall max: {global_max}\n")
    f.write(f"Mean: {mean}\n")
    f.write(f"Standard Deviation: {std}\n")

    print("\nGlobal statistics across all files:\n")
    print(f"Overall min: {global_min}\n")
    print(f"Overall max: {global_max}\n")
    print(f"Mean: {mean}\n")
    print(f"Standard Deviation: {std}\n")

print(f"結果已儲存至 {output_file}")
