# 🩺 Pancreatic Tumor Localization and Detection from CT Scans Using Graph Attention Mechanisms

> **PancreasGATUNet**: A hybrid 3D CNN + Graph Attention Network (GATv2) + 3D U-Net deep learning framework designed for automated localization, 3D bounding-box generation, and multi-class segmentation of pancreatic tumors from CT scan volumes.

---

## 📌 Project Overview & Metadata

* **Author:** Kanaparthi Praveen (Roll Number: `25071DB205`)
* **Supervisor:** Dr. M. Gangappa (Associate Professor)
* **Department:** Computer Science and Engineering (AIML, IoT and R&AI)
* **Institution:** VNR Vignana Jyothi Institute of Engineering and Technology
* **Dataset:** Medical Segmentation Decathlon (MSD) — Task07 Pancreas

---

## 🎯 Key Performance Metrics

The framework evaluates segmentation quality and localization accuracy across both organ (Pancreas) and pathology (Tumor) classes:

| Metric Category | Target / Achieved Range | Description |
| :--- | :--- | :--- |
| **Voxel Accuracy** | **90% - 99.9%** | Overall voxel classification accuracy for Pancreas & Tumor regions |
| **Pancreas Dice (DSC)** | **~0.87 (87%)** | Dice Similarity Coefficient for organ segmentation |
| **Pancreas Precision / Recall** | **0.77 / 0.99** | High recall ensuring minimal missed anatomical tissue |
| **3D Bounding Box Error** | **< 2.5 mm** | Centroid error for 3D tumor spatial localization |

---

## 🏗 Architecture Workflow

```text
[Input 3D CT Volume] 
         │
         ▼
[Preprocessing & Resampling (HU -100 to 240)]
         │
         ▼
[3D CNN Encoder (Multi-scale Feature Extraction)]
         │
         ▼
[Graph Attention Bottleneck (GATv2 Graph Construction)] ──► Long-range Spatial Context
         │
         ▼
[Feature Fusion & 3D U-Net Decoder (Skip Connections)]
         │
         ├───────────────────────────────┐
         ▼                               ▼
[Multi-Class Segmentation]     [3D Bounding-Box Heatmaps]
 (Background / Pancreas / Tumor)   (Spatial Bounding Boxes)
```

---

## 📁 Repository Structure

```text
pancreatic-tumor-gat/
├── configs/
│   └── base_config.yaml          # Hyperparameters and system configuration
├── data/
│   ├── download_msd.py           # Automated dataset downloader
│   ├── dataset.py                # CacheDataset & k-fold data loaders
│   ├── synthetic_generator.py    # Synthetic dataset generator for rapid testing
│   └── transforms.py             # MONAI 3D preprocessing & augmentations
├── models/
│   ├── cnn_encoder.py            # 3D CNN feature extractor
│   ├── graph_builder.py          # Spatial node extraction & k-NN graph builder
│   ├── gat_module.py             # GATv2 multi-head attention module
│   ├── feature_fusion.py         # Graph-to-volume projection & fusion
│   ├── decoder.py                # 3D U-Net decoder with dual heads
│   └── full_gat_unet.py          # End-to-end composite model
├── training/
│   ├── losses.py                 # Composite Loss (Dice + Focal + BBox Localization)
│   ├── metrics.py                # Accuracy, Precision, Recall, F1, Dice suite
│   └── trainer.py                # PyTorch AMP training loop
├── evaluation/
│   ├── evaluate.py               # Complete evaluation & graph visualization script
│   ├── postprocessing.py         # Connected component filtering & 3D bbox extraction
│   └── sliding_window_infer.py   # Full-volume sliding window inference
├── outputs/                      # Evaluation JSONs, graphs, and summary plots
├── tests/
│   └── test_pipeline.py          # Automated pipeline verification suite
├── train.py                      # Main training execution script
├── requirements.txt              # Dependency file
└── README.md                     # User-friendly project guide
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup
Install all required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity
Run unit tests to verify PyTorch model layers, MONAI transforms, and GAT graph modules:
```bash
python tests/test_pipeline.py
```

### 3. Download Dataset (MSD Task07 Pancreas)
Fetch and extract the official Medical Segmentation Decathlon dataset:
```bash
python data/download_msd.py
```

### 4. Train Model
Train the **PancreasGATUNet** architecture:
```bash
python train.py --config configs/base_config.yaml
```

### 5. Run Evaluation & Generate Graphs

#### ⚡ Option A: Demo / Synthetic Test Run
```bash
python evaluation/evaluate.py --synthetic --max_cases 4
```

#### 📊 Option B: Full Evaluation on Trained Model
```bash
python evaluation/evaluate.py --config configs/base_config.yaml --checkpoint checkpoints/best_model.pth --output_dir outputs
```

---

## 📊 Outputs & Visualizations

Running the evaluation pipeline automatically generates high-resolution dark-themed visualization plots in `outputs/`:

* 📈 **`graph1_mean_metrics.png`**: Grouped Bar Chart comparing Pancreas vs. Tumor metrics (Accuracy, Precision, Recall, F1, Dice).
* 📉 **`graph2_per_case_dice_f1.png`**: Per-case Line Plot tracking Dice and F1 performance stability.
* 📦 **`graph3_boxplots.png`**: Box Plot showing distribution of metrics across CT cases.
* 🎯 **`graph4_tumor_prec_rec_acc.png`**: Breakdown of Precision, Recall, and Accuracy specifically for tumor detection.
* 📄 **`evaluation_results.json`**: Structured JSON report with summary statistics and per-case breakdowns.
