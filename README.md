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
│   └── full_gat_unul.py          # End-to-end composite model
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
