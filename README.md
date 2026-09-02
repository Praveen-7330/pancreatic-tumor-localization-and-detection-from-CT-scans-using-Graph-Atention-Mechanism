# Pancreatic Tumor Localization and Detection from CT Scans Using Graph Attention Mechanisms

**Author:** Kanaparthi Praveen (Roll Number: 25071DB205)  
**Supervisor:** Dr. M. Gangappa (Associate Professor)  
**Department:** Computer Science and Engineering (AIML, IoT and R&AI)  
**Institution:** VNR Vignana Jyothi Institute of Engineering and Technology  

---

## 1. Project Abstract & Architecture

Pancreatic tumor detection from CT scans is challenging due to small tumor size, low contrast, irregular shape, and complex surrounding anatomy. This project proposes a Graph Attention-based deep learning framework (**PancreasGATUNet**) for pancreatic tumor localization and segmentation.

The architecture comprises:
- **Preprocessing Pipeline:** HU Windowing [-100, 240], isotropic resampling, and foreground/background balanced patch extraction.
- **3D CNN Encoder:** Dense multi-scale volumetric feature encoding with skip connections.
- **Graph Construction & GATv2 Attention:** Models relational dependencies across anatomical regions to disambiguate subtle tumor boundaries.
- **Feature Fusion & 3D Decoder:** Reconstructs high-resolution multi-class segmentation maps and extracts 3D tumor bounding-box localization coordinates.
- **Evaluation Suite:** Calculates Accuracy, Precision, Recall, F1 Score, Dice Similarity Coefficient (DSC), 3D HD95 distance, and 3D Bounding Box error on the Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset.

---

## 2. Directory Structure

```text
pancreatic-tumor-gat/
├── configs/
│   └── base_config.yaml          # Hyperparameters and settings
├── data/
│   ├── download_msd.py           # Automated dataset downloader
│   ├── dataset.py                # CacheDataset & k-fold data loaders
│   ├── synthetic_generator.py    # Synthetic dataset generator for testing
│   └── transforms.py             # MONAI 3D preprocessing & augmentations
├── models/
│   ├── cnn_encoder.py            # 3D CNN feature extractor
│   ├── graph_builder.py          # Spatial node extraction & k-NN graph builder
│   ├── gat_module.py             # GATv2 multi-head attention module
│   ├── feature_fusion.py         # Graph-to-volume projection & fusion
│   ├── decoder.py                # 3D U-Net decoder with dual heads
│   └── full_gat_unet.py          # End-to-end composite model
├── training/
│   ├── losses.py                 # Composite Loss (Dice + Focal + Localization)
│   ├── metrics.py                # Accuracy, Precision, Recall, F1, Dice suite
│   └── trainer.py                # PyTorch AMP training loop
├── evaluation/
│   ├── evaluate.py               # Complete evaluation & graph visualization script
│   ├── postprocessing.py         # Connected component filtering & 3D bbox extraction
│   └── sliding_window_infer.py   # Full-volume sliding window inference
├── tests/
│   └── test_pipeline.py          # Verification test suite
├── train.py                      # Main training execution script
├── requirements.txt              # Python dependency list
└── README.md                     # Project documentation
```

---

## 3. Step-by-Step Execution Guide

### Step 1: Environment Setup
Install all required packages:
```bash
pip install -r requirements.txt
```

### Step 2: Run Pipeline Verification Test
Verify all model layers, MONAI transformations, and PyTorch GAT modules:
```bash
python tests/test_pipeline.py
```

### Step 3: Download & Prepare Dataset
Download the official Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset:
```bash
python data/download_msd.py
```

### Step 4: Model Training
Train the PancreasGATUNet model:
```bash
python train.py --config configs/base_config.yaml
```

### Step 5: Model Evaluation & Graph Generation
Run the evaluation script to calculate Accuracy, Precision, Recall, F1 Score, Dice, HD95, and generate graph plots:

#### Option A: Quick Demonstration / Synthetic Test
```bash
python evaluation/evaluate.py --synthetic --max_cases 4
```

#### Option B: Full Evaluation on Trained Checkpoint
```bash
python evaluation/evaluate.py --config configs/base_config.yaml --checkpoint checkpoints/best_model.pth --output_dir outputs
```

---

## 4. Outputs & Results

Running the evaluation script generates:

1. **Console Summary Table:** Displays Accuracy, Precision, Recall, F1 Score, Dice, HD95, and Bounding Box error metrics for Pancreas and Tumor segmentation.
2. **Saved Evaluation JSON:** `outputs/evaluation_results.json` containing per-case and aggregated metric values.
3. **Graph Visualizations (Saved in `outputs/` folder):**
   - `graph1_mean_metrics.png` — Grouped Bar Chart of mean metrics.
   - `graph2_per_case_dice_f1.png` — Line Plots of per-case Dice and F1 scores.
   - `graph3_boxplots.png` — Box Plots of metric distributions across cases.
   - `graph4_tumor_prec_rec_acc.png` — Per-case Precision, Recall, and Accuracy breakdown for Tumor.
