Pancreatic Tumor Localization and Detection from CT Scans Using Graph Attention Mechanisms
Author: Kanaparthi Praveen (Roll Number: 25071DB205)
Supervisor: Dr. M. Gangappa (Associate Professor)
Department: Computer Science and Engineering (AIML, IoT and R&AI)
Institution: VNR Vignana Jyothi Institute of Engineering and Technology

1. Project Abstract & Architecture
Pancreatic tumor detection from CT scans is challenging due to small tumor size, low contrast, irregular shape, and complex surrounding anatomy. This project proposes a Graph Attention-based deep learning framework for pancreatic tumor localization and segmentation.

The architecture comprises:

Preprocessing Pipeline: HU Windowing [-100, 240], isotropic resampling, and foreground/background balanced patch extraction.
3D CNN Encoder: Dense multi-scale volumetric feature encoding with skip connections.
Graph Construction & GATv2 Attention: Models relational dependencies across anatomical regions to disambiguate subtle tumor boundaries.
Feature Fusion & 3D Decoder: Reconstructs high-resolution multi-class segmentation maps and extracts 3D tumor bounding-box localization coordinates.
Evaluation: Evaluated using Dice Similarity Coefficient (DSC), IoU, Precision, and Recall on the Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset.
2. Directory Structure
pancreas-gat-detection/
├── configs/
│   └── base_config.yaml          # Hyperparameters and settings
├── data/
│   ├── download_msd.py           # Automated dataset downloader
│   ├── dataset.py                # CacheDataset & k-fold data loaders
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
│   ├── metrics.py                # Dice, IoU, Precision, Recall suite
│   └── trainer.py                # PyTorch AMP training loop
├── evaluation/
│   ├── postprocessing.py         # Connected component filtering & 3D bbox extraction
│   └── sliding_window_infer.py   # Full-volume sliding window inference
├── tests/
│   └── test_pipeline.py          # Verification test suite
├── train.py                      # Main execution script
├── requirements.txt
└── README.md
3. Quickstart & Verification
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated pipeline verification
python tests/test_pipeline.py

# 3. Download MSD Task07 dataset
python data/download_msd.py

**Author:** Kanaparthi Praveen (Roll Number: 25071DB205)  
**Supervisor:** Dr. M. Gangappa (Associate Professor)  
**Department:** Computer Science and Engineering (AIML, IoT & AI)  
**Institution:** VNR Vignana Jyothi Institute of Engineering and Technology

---

## 1. Project Abstract & Architecture
Pancreatic tumor detection from CT scans is challenging due to small tumor size, low contrast, irregular shape, and complex surrounding anatomy. This project proposes a **Graph Attention‑based deep learning framework** for pancreatic tumor localization and segmentation.

The architecture comprises:
- **Preprocessing Pipeline**: HU Windowing `[-100, 240]`, isotropic resampling, and balanced patch extraction.
- **3D CNN Encoder**: Dense multi‑scale volumetric feature encoding with skip connections.
- **Graph Construction & GATv2 Attention**: Models relational dependencies across anatomical regions to disambiguate subtle tumor boundaries.
- **Feature Fusion & 3D Decoder**: Reconstructs high‑resolution multi‑class segmentation maps and extracts 3D tumor bounding‑box coordinates.
- **Evaluation**: Dice Similarity Coefficient (DSC), IoU, Precision, and Recall on the **Medical Segmentation Decathlon (MSD) Task07 Pancreas** dataset.

---

## 2. Directory Structure
```
pancreatic-tumor-gat/
├── configs/                 # Hyper‑parameters and settings
├── data/                    # Dataset utilities
│   ├── download_msd.py       # Automated dataset downloader (now OS‑agnostic)
│   ├── dataset.py            # CacheDataset & k‑fold data loaders
│   └── synthetic_generator.py # Synthetic benchmark dataset generator
├── models/                  # Model components
│   └── full_gat_unet.py      # End‑to‑end composite model
├── training/                # Training utilities
│   ├── trainer.py            # PyTorch AMP training loop
│   └── ...
├── evaluation/              # Inference & post‑processing
├── utils/                   # **New utilities**
│   ├── device_utils.py       # Cross‑platform device selection (CUDA/MPS/CPU)
│   └── path_utils.py         # OS‑agnostic repository‑root path resolution
├── tests/                   # Verification test suite
├── train.py                  # Main execution script (now uses utils)
├── requirements.txt
└── README.md
```

---

## 3. Universal Enhancements
- **Cross‑platform device selection** via `utils/device_utils.get_device()` (CUDA → MPS → CPU).
- **OS‑agnostic path handling** using `utils.path_utils.resolve_path(...)` – works on Windows, macOS, and Linux without manual path changes.
- All hard‑coded paths have been replaced, making the project portable.

---

## 4. Quickstart & Verification
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify the pipeline with synthetic data (fast, no download)
python tests/test_pipeline.py

# 3. Run a dry‑run training on synthetic data
python train.py --synthetic --epochs 2

# 4. Download the full MSD Task07 dataset (once)
python data/download_msd.py

# 5. Train on the full dataset (adjust epochs as needed)
python train.py --epochs 100
```
The script automatically selects the best available device and resolves all paths, so you can run it on any operating system.

---

## 5. Further Usage
- To change the data root, pass `--data_root <path>` (future extension).
- The project remains at a **CSE master’s‑level scope** – no unnecessary complexity beyond the graph‑attention approach.

---

*Feel free to explore, modify, and extend the graph attention mechanisms for other medical imaging tasks.*

**Author:** Kanaparthi Praveen (Roll Number: 25071DB205)  
**Supervisor:** Dr. M. Gangappa (Associate Professor)  
**Department:** Computer Science and Engineering (AIML, IoT and R&AI)  
**Institution:** VNR Vignana Jyothi Institute of Engineering and Technology  

---

## 1. Project Abstract & Architecture
Pancreatic tumor detection from CT scans is challenging due to small tumor size, low contrast, irregular shape, and complex surrounding anatomy. This project proposes a **Graph Attention-based deep learning framework** for pancreatic tumor localization and segmentation.

The architecture comprises:
- **Preprocessing Pipeline**: HU Windowing `[-100, 240]`, isotropic resampling, and foreground/background balanced patch extraction.
- **3D CNN Encoder**: Dense multi-scale volumetric feature encoding with skip connections.
- **Graph Construction & GATv2 Attention**: Models relational dependencies across anatomical regions to disambiguate subtle tumor boundaries.
- **Feature Fusion & 3D Decoder**: Reconstructs high-resolution multi-class segmentation maps and extracts 3D tumor bounding-box localization coordinates.
- **Evaluation**: Evaluated using Dice Similarity Coefficient (DSC), IoU, Precision, and Recall on the **Medical Segmentation Decathlon (MSD) Task07 Pancreas** dataset.

---

## 2. Directory Structure
```
pancreas-gat-detection/
├── configs/
│   └── base_config.yaml          # Hyperparameters and settings
├── data/
│   ├── download_msd.py           # Automated dataset downloader
│   ├── dataset.py                # CacheDataset & k-fold data loaders
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
│   ├── metrics.py                # Dice, IoU, Precision, Recall suite
│   └── trainer.py                # PyTorch AMP training loop
├── evaluation/
│   ├── postprocessing.py         # Connected component filtering & 3D bbox extraction
│   └── sliding_window_infer.py   # Full-volume sliding window inference
├── tests/
│   └── test_pipeline.py          # Verification test suite
├── train.py                      # Main execution script
├── requirements.txt
└── README.md
```

---

## 3. Quickstart & Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated pipeline verification
python tests/test_pipeline.py

# 3. Download MSD Task07 dataset
python data/download_msd.py
```
