# Pancreatic Tumor Localization and Detection from CT Scans Using Graph Attention Mechanisms

**Kanaparthi Praveen**  
Roll Number: 25071DB205  
Department of Computer Science and Engineering (AIML, IoT and R&AI)  
VNR Vignana Jyothi Institute of Engineering and Technology, Hyderabad, Telangana, India  
*Email:* praveen.25071DB205@vnrvjiet.in  

**Dr. M. Gangappa**  
Associate Professor, Department of Computer Science and Engineering  
VNR Vignana Jyothi Institute of Engineering and Technology, Hyderabad, Telangana, India  

---

## Abstract

Accurate localization and multi-class volumetric segmentation of pancreatic tumors from 3D abdominal Computed Tomography (CT) scans remain significantly challenging tasks due to the small size of tumor targets, low soft-tissue attenuation contrast, irregular morphological boundaries, and complex surrounding retroperitoneal anatomy. Conventional 3D Convolutional Neural Networks (3D CNNs) rely strictly on local spatial kernel operations, making them incapable of modeling long-range contextual relationships across distant anatomical regions. To overcome these limitations, this paper proposes **PancreasGATUNet**, a novel hybrid deep learning architecture combining a multi-scale 3D CNN encoder-decoder with a multi-head **GATv2 Graph Attention Mechanism** embedded within the bottleneck representation layer. The framework constructs spatial $k$-nearest neighbor ($k$-NN) graph representations over sub-volumetric feature grids and applies dynamic graph attention to adaptively re-weight contextual dependencies between tumor-organ interfaces. Furthermore, a dual-head decoder joint-optimizes voxel-level segmentation alongside 3D bounding-box localization coordinates using a composite multi-task loss function combining Dice Loss, Focal Loss ($\gamma=2.0$), and Smooth L1 localization loss. Evaluated on the Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset, the proposed framework achieves a **78.40% Dice Similarity Coefficient (DSC)** and an **81.20% Precision** for pancreatic tumor segmentation, alongside a 95th Percentile Hausdorff Distance (HD95) of **12.40 mm** and a 3D bounding-box centroid distance error of **14.20 mm**. Experimental results demonstrate a **+10.2% improvement in Tumor Dice score** over conventional 3D U-Net baselines, proving the clinical viability of graph attention mechanisms for soft-tissue abdominal oncology.

**Keywords:** Medical Image Segmentation, Pancreatic Tumor Detection, Graph Attention Networks (GATv2), 3D CT Scans, Bounding Box Localization, Deep Learning in Healthcare.

---

## 1. Introduction

Pancreatic cancer is one of the deadliest gastrointestinal malignancies worldwide, characterized by an exceptionally low five-year survival rate of less than 11%. Early and precise radiological detection on Computed Tomography (CT) scans is critical for surgical resection planning and patient prognosis. However, automated computer-aided diagnosis (CAD) of pancreatic lesions presents four formidable technical bottlenecks:

1. **Extreme Volumetric Class Imbalance:** The pancreas organ occupies $< 0.5\%$ of the total abdominal CT scan volume, while tumor tissue often accounts for $< 0.05\%$, leading to severe background-dominated voxel bias during neural network optimization.
2. **Low Attenuation Contrast:** Pancreatic tumor tissue exhibits Hounsfield Unit (HU) radiodensity values ($30 - 70 	ext{ HU}$) that overlap substantially with healthy parenchymal tissue and adjacent retroperitoneal fat or blood vessels.
3. **Morphological Variability:** Tumors lack rigid geometric shapes, presenting highly irregular, ill-defined infiltrative boundaries.
4. **Receptive Field Limitations of Convolutions:** Standard 3D convolutions operate locally using small spatial kernels ($3 	imes 3 	imes 3$). Consequently, local operations fail to capture long-range anatomical dependencies across non-local organs (e.g., duodenum, spleen, stomach, aorta).

While deep learning architectures such as 3D U-Net and 3D V-Net have achieved success in liver and kidney segmentation, their reliance on local spatial convolutions leads to boundary smoothing and false-positive tumor predictions in surrounding tissues. Self-attention mechanisms such as 3D Vision Transformers (ViTs) attempt to address long-range dependencies; however, they require prohibitively high memory overhead ($O(N^2)$ quadratic complexity) when applied to high-resolution 3D medical volumes.

To address these challenges, this study presents **PancreasGATUNet**, a lightweight yet powerful hybrid architecture that integrates **Graph Attention Networks (GATv2)** into a 3D U-Net feature bottleneck. By representing downsampled volumetric sub-regions as spatial nodes in a topological graph, GATv2 computes dynamic attention weights between anatomical regions without incurring the quadratic computational cost of full volumetric transformers.

### Main Contributions:
- **Novel Hybrid Architecture (`PancreasGATUNet`):** Integrates 3D spatial node extraction and multi-head GATv2 graph attention into a 3D U-Net bottleneck to model long-range spatial context for soft-tissue boundary disambiguation.
- **Dual-Head Multi-Task Learning:** Simultaneously performs multi-class voxel segmentation (Background, Pancreas Organ, Tumor) and automated 3D Bounding-Box coordinate regression.
- **Composite Multi-Task Loss:** Combines Dice Loss, Focal Loss ($\gamma=2.0$), and Smooth L1 Loss to tackle severe class imbalance and penalize localization errors.
- **Comprehensive Empirical Validation:** Extensive evaluation on the Medical Segmentation Decathlon (MSD) Task07 Pancreas benchmark, outperforming conventional 3D U-Net and 3D Attention U-Net baselines across 8 quantitative metrics.

---

## 2. Related Work

### 2.1 3D Volumetric Medical Segmentation
Ronneberger et al. introduced the U-Net architecture for 2D biomedical image segmentation, which was later extended to 3D by Çiçek et al. (3D U-Net) and Milletari et al. (V-Net). These architectures utilize encoder-decoder structures with skip connections to preserve high-resolution spatial details. MONAI framework abstractions have further standardized 3D sliding-window inference and spatial augmentations. However, purely convolutional encoders remain limited by local spatial receptive fields.

### 2.2 Attention Mechanisms & Graph Neural Networks
Oktay et al. proposed Attention U-Net, introducing additive spatial attention gates to highlight salient features in skip connections. Vaswani et al. introduced self-attention in Transformers, which was adapted to 3D medical imaging by Hatamizadeh et al. (UNETR, Swin UNETR). Although effective, 3D Transformers suffer from extreme memory footprints during training.

Graph Neural Networks (GNNs) present a memory-efficient alternative by modeling data as graph nodes and edges. Veličković et al. introduced Graph Attention Networks (GAT), enabling nodes to assign anisotropic weights to neighbors. Brody et al. developed **GATv2**, fixing the static attention problem in original GAT by expressing dynamic attention query-dependency. In this work, we leverage GATv2 over 3D spatial sub-volume node grids.

---

## 3. Methodology & System Architecture

The overall pipeline of the proposed **PancreasGATUNet** framework is illustrated in Figure 1. The architecture consists of four primary stages: (1) Medical Preprocessing, (2) 3D CNN Volumetric Encoder, (3) Spatial Node Graph Construction & GATv2 Attention Bottleneck, and (4) Dual-Head 3D Decoder.

```
       +--------------------------------------------------------+
       |   Input 3D CT Volume [1, D, H, W] (HU Windowed)        |
       +---------------------------+----------------------------+
                                   |
                                   v
       +--------------------------------------------------------+
       |   3D CNN Volumetric Encoder (4 Resolution Levels)      |
       +---------------------------+----------------------------+
                                   |
                                   v
       +--------------------------------------------------------+
       |   Spatial Node Grid Construction (4x4x4 = 64 Nodes)    |
       |   & k-NN Spatial Topology Graph Construction (k=8)    |
       +---------------------------+----------------------------+
                                   |
                                   v
       +--------------------------------------------------------+
       |   GATv2 Multi-Head Dynamic Graph Attention Module      |
       +---------------------------+----------------------------+
                                   |
                                   v
       +--------------------------------------------------------+
       |   Graph-to-Volume Projection & Skip Connection Fusion  |
       +---------------------------+----------------------------+
                                   |
                                   v
       +--------------------------------------------------------+
       |                 Dual-Head 3D Decoder                   |
       |  +------------------------+  +----------------------+  |
       |  | Voxel Segmentation Head|  | 3D BBox Localizer    |  |
       |  +------------------------+  +----------------------+  |
       +--------------------------------------------------------+
```
*Figure 1: End-to-end architecture flow of the proposed PancreasGATUNet framework.*

### 3.1 Medical Image Preprocessing Pipeline
Raw abdominal CT volumes exhibit extreme intensity range variations ($-1000 	ext{ HU}$ to $+3000 	ext{ HU}$). The preprocessing pipeline applies:
1. **Hounsfield Unit (HU) Windowing:** Intensities are clipped to the range $[-100, 240] 	ext{ HU}$ to eliminate uninformative bone and air voxels while boosting soft-tissue contrast:
   $$I_{	ext{win}}(x,y,z) = \min(\max(I(x,y,z), -100), 240)$$
2. **Isotropic Resampling:** Volumes are resampled using 3D spline interpolation to an isotropic spatial resolution of $1.0 	imes 1.0 	imes 1.0 	ext{ mm}^3$ per voxel.
3. **Intensity Normalization:** Z-score normalization scales voxel intensities to zero mean and unit variance:
   $$I_{	ext{norm}} = rac{I_{	ext{win}} - \mu}{\sigma}$$
4. **Foreground-Balanced Patch Sampling:** Random sub-volume patches of size $96 	imes 96 	imes 96$ voxels are cropped during training, with a $50\%$ probability centered on positive organ/tumor voxels.

### 3.2 3D CNN Volumetric Encoder
The encoder consists of 4 sequential downsampling blocks. Each block comprises two $3 	imes 3 	imes 3$ 3D convolutions, Instance Normalization, and LeakyReLU activation ($lpha=0.01$), followed by a $2 	imes 2 	imes 2$ max-pooling operation. Feature channel dimensions expand sequentially across levels: $C \in \{16, 32, 64, 128\}$. For an input volume patch $X \in \mathbb{R}^{1 	imes 96 	imes 96 	imes 96}$, the bottleneck outputs a feature map $F_b \in \mathbb{R}^{128 	imes 12 	imes 12 	imes 12}$.

### 3.3 Spatial Node Construction & k-NN Graph Builder
To convert the continuous bottleneck feature map $F_b$ into a graph representation:
1. **Spatial Grid Partitioning:** $F_b$ is partitioned into a 3D spatial grid of $G_z 	imes G_y 	imes G_x = 4 	imes 4 	imes 4 = 64$ sub-volumetric nodes.
2. **Feature Aggregation:** Spatial average pooling condenses each grid cell into a node feature vector $\mathbf{h}_i \in \mathbb{R}^{128}$ ($i = 1, \dots, 64$).
3. **$k$-NN Topology Construction:** Edges $\mathcal{E}$ are constructed between nodes using Euclidean spatial distance between 3D grid centroids. Each node is connected to its $k=8$ nearest spatial neighbors, creating an adjacency matrix $\mathbf{A} \in \mathbb{R}^{N 	imes N}$.

### 3.4 GATv2 Multi-Head Graph Attention Mechanism
The constructed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is passed through a multi-head GATv2 layer. GATv2 computes dynamic attention coefficients $lpha_{ij}$ measuring the relative importance of node $j$ to node $i$:

$$lpha_{ij} = rac{\exp\left(\mathbf{a}^T 	ext{LeakyReLU}\left(\mathbf{W} [\mathbf{h}_i \,||\, \mathbf{h}_j]ight)ight)}{\sum_{k \in \mathcal{N}_i} \exp\left(\mathbf{a}^T 	ext{LeakyReLU}\left(\mathbf{W} [\mathbf{h}_i \,||\, \mathbf{h}_k]ight)ight)}$$

where $\mathbf{W} \in \mathbb{R}^{F' 	imes F}$ is a shared linear transformation matrix, $\mathbf{a} \in \mathbb{R}^{F'}$ is a learnable attention vector, and $||$ denotes vector concatenation.

With $M=2$ independent attention heads, the updated node representation $\mathbf{h}'_i$ is computed via multi-head aggregation:
$$\mathbf{h}'_i = \|_{m=1}^M \sigma \left( \sum_{j \in \mathcal{N}_i} lpha_{ij}^m \mathbf{W}^m \mathbf{h}_j ight)$$

The updated node embeddings are projected back to a 3D volumetric feature tensor $F_{	ext{gat}} \in \mathbb{R}^{128 	imes 12 	imes 12 	imes 12}$ and added to the original bottleneck representation via residual connection:
$$F_{	ext{out}} = F_b + F_{	ext{gat}}$$

### 3.5 Dual-Head 3D Decoder & Composite Loss Function
The decoder mirrors the encoder, utilizing 3D transposed convolutions for spatial upsampling and skip connections to merge high-resolution encoder features. The architecture splits into two output heads:
- **Segmentation Head:** Outputs a 3-channel logits tensor $S \in \mathbb{R}^{3 	imes D 	imes H 	imes W}$ evaluated via Softmax for class probabilities (0: Background, 1: Pancreas, 2: Tumor).
- **Localization Head:** Predicts 3D Gaussian heatmap distributions centered at the tumor centroid and regresses bounding box offsets $(z_{\min}, z_{\max}, y_{\min}, y_{\max}, x_{\min}, x_{\max})$.

The model is trained end-to-end using a Composite Multi-Task Loss:
$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{Dice}} + \lambda_1 \mathcal{L}_{	ext{Focal}} + \lambda_2 \mathcal{L}_{	ext{BBox}}$$

$$\mathcal{L}_{	ext{Dice}} = 1 - rac{2 \sum_{c=1}^2 \sum_v p_{c,v} g_{c,v} + \epsilon}{\sum_{c=1}^2 \sum_v (p_{c,v}^2 + g_{c,v}^2) + \epsilon}$$

$$\mathcal{L}_{	ext{Focal}} = -rac{1}{V} \sum_{v=1}^V \sum_{c=1}^2 (1 - p_{c,v})^\gamma \log(p_{c,v})$$

where $\gamma = 2.0$, $\lambda_1 = 1.0$, and $\lambda_2 = 0.5$.

---

## 4. Experimental Setup & Implementation Details

### 4.1 Dataset & Evaluation Protocols
The model was trained and evaluated on the **Medical Segmentation Decathlon (MSD) Task07 Pancreas** dataset, consisting of 281 abdominal contrast-enhanced 3D CT scans. A 5-fold cross-validation scheme was employed (80% training, 20% validation).

### 4.2 Hyperparameters & Training Environment
- **Framework:** PyTorch 2.1 + MONAI 1.3
- **Optimizer:** AdamW ($	ext{lr} = 1	imes 10^{-4}$, weight decay $= 1	imes 10^{-5}$)
- **Learning Rate Schedule:** Cosine Annealing LR ($T_{\max} = 100$ epochs)
- **Batch Size:** 2 patches per GPU using Automatic Mixed Precision (AMP FP16)
- **Hardware Acceleration:** Executed via dynamic device router supporting NVIDIA CUDA, Apple Silicon MPS, and CPU backends.

---

## 5. Experimental Results & Analysis

### 5.1 Quantitative Performance Comparison
Table 1 outlines the performance of **PancreasGATUNet** compared against benchmark architectures on the MSD Task07 validation set.

*Table 1: Quantitative segmentation performance comparison on MSD Task07 Pancreas dataset (Mean values across 5 folds).*

| Architecture | Target Class | Dice (DSC) % ↑ | IoU % ↑ | Precision % ↑ | Recall % ↑ | F1 Score ↑ | Accuracy % ↑ | HD95 (mm) ↓ | Centroid Error (mm) ↓ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard 3D U-Net** | Pancreas | 81.40% | 68.60% | 71.20% | 95.40% | 0.8140 | 92.10% | 58.20 mm | N/A |
| | Tumor | 68.20% | 51.70% | 72.40% | 64.50% | 0.6820 | 92.40% | 22.80 mm | 28.40 mm |
| **3D Attention U-Net**| Pancreas | 84.10% | 72.50% | 74.10% | 97.80% | 0.8410 | 93.60% | 52.10 mm | N/A |
| | Tumor | 72.90% | 57.35% | 76.80% | 69.40% | 0.7290%| 93.80% | 17.50 mm | 19.80 mm |
| **PancreasGATUNet (Ours)**| **Pancreas** | **86.82%** | **76.71%** | **76.81%** | **99.83%** | **0.8682** | **94.77%** | **47.67 mm** | N/A |
| | **Tumor** | **78.40%** | **65.10%** | **81.20%** | **76.10%** | **0.7840** | **94.80%** | **12.40 mm** | **14.20 mm** |

### 5.2 Key Findings & Discussion
1. **Tumor Segmentation Gain:** The proposed model achieves a **+10.2% increase in Tumor Dice score** ($78.40\%$ vs. $68.20\%$) over the standard 3D U-Net baseline.
2. **Boundary Disambiguation (HD95 Reduction):** The 95th Percentile Hausdorff Distance drops from $22.80 	ext{ mm}$ to **$12.40 	ext{ mm}$**, confirming that GATv2 attention effectively suppresses boundary ambiguity at tumor-parenchyma interfaces.
3. **High Tumor Precision (81.20%):** Spatial graph attention prevents false-positive voxel assignments in non-pancreatic soft tissues.
4. **3D Bounding-Box Accuracy:** The dual-head localizer achieves a **14.20 mm centroid distance error** and a 3D Bounding Box IoU of **0.7100**, enabling reliable automated ROI detection.

---

## 6. Conclusion & Future Work

This paper presented **PancreasGATUNet**, a deep learning framework combining a 3D CNN encoder-decoder with a bottleneck GATv2 graph attention mechanism for pancreatic tumor localization and segmentation from 3D CT scans. By representing sub-volumetric feature spaces as topological graphs, GATv2 models long-range relational dependencies across anatomical regions while avoiding the prohibitive memory constraints of 3D Vision Transformers. Experimental evaluation on the MSD Task07 Pancreas benchmark demonstrates significant performance improvements across 8 metrics, achieving a **78.40% Tumor Dice score** and a **14.20 mm 3D localization accuracy**.

**Future Scope:**
- Expanding graph topologies to dynamic multi-scale graphs across all encoder resolution levels.
- Integrating multimodal data (e.g., paired CT and MRI volumes).
- Clinical deployment for automated surgical margin planning.

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 234-241.
2. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. *MICCAI*, 424-432.
3. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*.
4. Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? *ICLR*.
5. Oktay, O., Schlemper, J., Folgoc, L. L., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *arXiv preprint arXiv:1804.03999*.
6. Antonelli, M., Reinke, A., Bakas, S., et al. (2022). The Medical Segmentation Decathlon. *Nature Communications*, 13(1), 4128.
7. Hatamizadeh, A., Nath, V., Tang, Y., et al. (2022). UNETR: Transformers for 3D medical image segmentation. *WACV*, 574-584.
