# Data Preprocessing

---

## **Step 0: `Step0.py`**

Extracts 3D cubes (e.g., 32×32×32) centered on annotated neuron coordinates. Combines raw and masked voxel channels.

* **Input**: `.raw` volumes + `.txt` annotation files
* **Output**: 2-channel `.npy` cubes
* **Key features**: Robust padding, coordinate alignment, optional `.raw` or `.npy` output
* Key Reference: `Coordinate_of_Cell`
* Download: 
  * [Crop Raw](https://drive.google.com/file/d/1GUNUqphCDl55A1IlVHcDgbF0k1Ndpqou/view?usp=drive_link)
  * [Masked](https://drive.google.com/file/d/1tO4FY32HbxaplBzLRTMPqHRc7pMYPE_r/view?usp=drive_link)

NOTE: you can only crop raw Cube from the original worm data. (Skip the concatenation of Mask section)

NOTE: If you only need original data (without mask), you can just skip Step1.py and Step2.py (directly run Merge.py right after Step0.py)

---

## **Step 1: `Step1.py`**

Adds a **soft boundary mask** (third channel) using a sigmoid-transformed distance from labeled regions.

* **Input**: 2-channel cubes
* **Output**: 3-channel cubes (`[raw, binary_mask, soft_mask]`)
* **Method**: Applies a decaying weight outside label boundaries

---

## **Step 2: `Step2.py`**

Extracts only the soft-masked voxel (`3rd channel`) for direct use in model training.

* **Input**: 3-channel cubes
* **Output**: 1-channel soft-masked `.npy` cubes
* **Purpose**: Simplifies dataset for downstream models

---

## **Merge: `Merge.py`**

Merges all individual cubes per worm into a single 5D array for efficient batched access.

* **Shape**: `(558, 1, 32, 32, 32)` per worm
* **Output**: One `.npy` file per worm
* **Note**: Automatically fills missing cells with zeroed cubes

---

## **Validation: `TestSize.py`**

Scans the merged dataset and identifies any `.npy` files with incorrect shapes or loading issues.

* **Output**: List of problematic files, if any

---

## **Visualization: `VisMergedData.py`**

Visualizes 3 random cells from a random worm, displaying:

* XY, XZ, YZ mid-slices

* Interactive colormap support (`viridis`, `inferno`, etc.)

* **Output**: Interactive matplotlib plot

---

## ✅ Recommended Workflow

```bash
# 1. Extract 2-channel cubes from raw + mask volumes
python Step0.py

# 2. Add soft boundary as 3rd channel
python Step1.py

# 3. Extract soft-masked cube for training
python Step2.py

# 4. Merge all cubes per worm
python Merge.py

# 5. Check for invalid files
python TestSize.py

# 6. Visualize samples
python VisMergedData.py
```

---

## 📦 Output Directory Structure

```
├── 2ChannelMaskedCube32/      # After Step0
├── 3ChannelMaskedCube32/      # After Step1
├── MaskedCube32/              # After Step2
├── MergedCubes32/             # After Merge
├── skipped_files.txt          # From Step1
├── processed_files.txt        # From Step1
```
---

# Some Explanation of other Reference files or folders

## 🧬 `Coordinates of cube.zip`

### 🔹 Folder: `Coordinate_of_Cell/`

This directory contains `.txt` files with **annotated neuron coordinates** for each worm, used as inputs in `Step0.py`.

### 📄 File Format: `worm_###.txt`

Each file corresponds to a single worm and contains tab-separated information about its identified neurons. The key columns include:

| Column         | Description                                                |
| -------------- | ---------------------------------------------------------- |
| `label_number` | Unique numeric ID for the neuron                           |
| `label_name`   | Neuron name (e.g., `ADAL`, `PHAR`, `RIGL`, etc.)           |
| `x`, `y`, `z`  | 3D coordinates of the neuron center (float precision)      |
| `Aligned_No`   | A unique sequential index used as the identifier in output |

### 🔍 Example Row:

```
label_number	label_name	    x	            y	          z	        Aligned_No
...
195	        ADAL	            1242.000000	    84.000000	  33.000000	1
...
```

* `ADAL` is the neuron label.
* The position is at `(x=1242, y=84, z=33)`.
* This is the first aligned neuron in the list.

### 🧪 Usage in Step0

In `Step0.py`, these files are loaded via:

```python
txt_path = os.path.join(txt_dir, txt_file)
coordinates = read_all_coordinates(txt_path)
```

Each row provides a target for extracting a **32×32×32 cube** around the neuron center. The extracted cube is named as:

```
<worm_name>_cube_<Aligned_No>.npy
```