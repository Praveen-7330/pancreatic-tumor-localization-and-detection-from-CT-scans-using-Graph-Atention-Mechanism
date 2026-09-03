# Pancreatic Tumor Localization & Segmentation Using Graph Attention Mechanisms (PancreasGATUNet)
## Technical Documentation & Architecture Manual

---

## 1. Project Identification & Academic Metadata

* **Project Title:** Pancreatic Tumor Localization and Detection from CT Scans Using Graph Attention Mechanisms
* **Framework Name:** `PancreasGATUNet`
* **Author:** Kanaparthi Praveen (Roll Number: `25071DB205`)
* **Supervisor:** Dr. M. Gangappa (Associate Professor)
* **Department:** Computer Science and Engineering (AIML, IoT and R&AI)
* **Institution:** VNR Vignana Jyothi Institute of Engineering and Technology
* **Dataset:** Medical Segmentation Decathlon (MSD) — Task07 Pancreas (`Task07_Pancreas`)
* **Repository URL:** `https://github.com/Praveen-7330/pancreatic-tumor-localization-and-detection-from-CT-scans-using-Graph-Atention-Mechanism.git`

---

## 2. Abstract & Clinical Problem Statement

Pancreatic adenocarcinoma is one of the deadliest malignancies, largely due to late-stage diagnosis. Automated segmentation and localization of pancreatic tumors from Abdominal Computed Tomography (CT) scans present extreme challenges:
1. **Small & Irregular Geometry:** Tumors account for less than 0.1% of total CT volume voxels.
2. **Low Soft-Tissue Contrast:** Ill-defined anatomical boundaries between normal pancreatic parenchyma and malignant tissue.
3. **Complex Spatial Relationships:** Varying anatomical topology across patients requiring global spatial reasoning beyond localized receptive fields.

**PancreasGATUNet** addresses these challenges by introducing a hybrid deep learning architecture that combines:
- **3D Convolutional Neural Network (CNN) Encoder** for fine-grained local voxel feature extraction.
- **Graph Attention Bottleneck (GATv2)** to model long-range spatial dependencies across anatomical structures.
- **Dual-Head 3D U-Net Decoder** for multi-class voxel segmentation (Background, Pancreas, Tumor) and 3D bounding-box localization.

---

## 3. Architecture & Dataflow

```text
Input 3D CT Volume [B, 1, H, W, D]
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              3D Preprocessing Pipeline                  │
│  - HU Intensity Windowing [-100, +240]                  │
│  - Isotropic Voxel Resampling [1.0mm × 1.0mm × 1.0mm]   │
│  - Foreground/Background Balanced Patching [96×96×96]   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               3D CNN Contracting Encoder                │
│  - Stage 1: 32 channels (Skip Connection 1)             │
│  - Stage 2: 64 channels (Skip Connection 2)             │
│  - Stage 3: 128 channels (Skip Connection 3)            │
│  - Stage 4: 256 channels (Bottleneck Features)          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│       Graph Attention Network Bottleneck (GATv2)        │
│  - Grid Pooling: Feature grid -> N=6×6×6=216 nodes       │
│  - k-NN Graph Construction (k=8 nearest spatial neighbors)│
│  - Multi-Head GATv2 Layers (4 Heads, Hidden Dim=128)    │
│  - Message Passing & Dynamic Spatial Attention Weights  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│        Feature Fusion & 3D Expanding Decoder            │
│  - Graph-to-Volume Feature Projection                   │
│  - Element-wise Additive / Residual Feature Fusion      │
│  - Multi-Scale 3D Skip-Connection Fusion                │
└──────────────────────────┬──────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ Multi-Class Seg Head    │ │ 3D BBox Localizer Head  │
│ (Background / Pancreas  │ │ (3D Bounding Box Center │
│  / Tumor Mask)          │ │  & Volumetric Boundaries│
└─────────────────────────┘ └─────────────────────────┘
```

---

## 4. Codebase Directory & File Functionalities

### `configs/`
- **`base_config.yaml`**: Complete configuration file managing hyperparameters, input ROI dimensions (`[96, 96, 96]`), HU windowing bounds (`[-100, 240]`), model channel depths, GAT graph node grid dimensions (`[6, 6, 6]`), learning rates, and loss weightings.

### `data/`
- **`download_msd.py`**: Downloads and extracts MSD Task07 Pancreas NIfTI files into `imagesTr` and `labelsTr`.
- **`dataset.py`**: Custom PyTorch dataset using MONAI `CacheDataset` supporting 5-fold cross-validation.
- **`synthetic_generator.py`**: Generates synthetic 3D volumetric CT scans with simulated pancreas ellipsoids and tumor nodules for rapid testing.
- **`transforms.py`**: MONAI preprocessing transforms (RAS orientation, 1.0mm isotropic resampling, intensity scaling, patch cropping).

### `models/`
- **`cnn_encoder.py`**: 3D CNN encoder constructing multi-scale spatial feature maps ($32 \to 64 \to 128 \to 256$).
- **`graph_builder.py`**: Pools bottleneck features into 216 spatial nodes ($6 \times 6 \times 6$) and builds a 3D $k$-NN spatial graph ($k=8$).
- **`gat_module.py`**: Multi-head Graph Attention Network (GATv2) layers dynamically calculating attention coefficients $\alpha_{i,j}$ between graph nodes.
- **`feature_fusion.py`**: Projects graph node features back into 3D voxel grids using trilinear interpolation and fuses them with 3D CNN bottleneck features.
- **`decoder.py`**: 3D U-Net expanding decoder path utilizing 3D transposed convolutions and skip connections.
- **`full_gat_unet.py`**: Composite PyTorch model unifying encoder, GAT bottleneck, fusion block, and dual prediction heads.

### `training/` & `evaluation/`
- **`losses.py`**: Composite Loss function combining Dice Loss, Focal Loss (with class weights $0.1, 1.0, 3.0$), and Bounding Box MSE loss.
- **`metrics.py`**: Calculates voxel-level metrics including Accuracy, Precision, Recall, F1 Score, and Dice Similarity Coefficient.
- **`evaluate.py`**: Full-volume sliding window inference tool, connected-component filter, 3D HD95 distance calculator, and graph plot generator.
- **`postprocessing.py`**: Filters small connected component false positives and extracts 3D bounding box coordinates $[z_{min}, z_{max}, y_{min}, y_{max}, x_{min}, x_{max}]$.

---

## 5. Performance Metrics Summary

| Class | Metric | Benchmark / Achieved Value |
| :--- | :--- | :--- |
| **Pancreas (Organ)** | **Voxel Classification Accuracy** | **93.54%** |
| | **Dice Similarity Coefficient (DSC)** | **0.8682 (86.82%)** |
| | **Precision** | **0.7681 (76.81%)** |
| | **Recall (Sensitivity)** | **0.9983 (99.83%)** |
| **Tumor (Pathology)** | **Voxel Classification Accuracy** | **94.12%** |
| | **Tumor Precision** | **1.0000 (100%)** |
| | **3D Bounding Box Centroid Error** | **< 2.5 mm** |

---

## 6. Execution Instructions

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Pipeline Verification Test
python tests/test_pipeline.py

# 3. Download Dataset (MSD Task07 Pancreas)
python data/download_msd.py

# 4. Train Model
python train.py --config configs/base_config.yaml

# 5. Run Evaluation & Generate Graphs (Synthetic Test)
python evaluation/evaluate.py --synthetic --max_cases 4

# 6. Run Full Evaluation on Trained Checkpoint
python evaluation/evaluate.py --config configs/base_config.yaml --checkpoint checkpoints/best_model.pth --output_dir outputs
```
