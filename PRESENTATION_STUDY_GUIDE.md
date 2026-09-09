# Master's Final Presentation & Viva Prep Guide

**Project Title:** Pancreatic Tumor Localization and Detection from CT Scans Using Graph Attention Mechanisms  
**Student Name:** Kanaparthi Praveen (Roll Number: 25071DB205)  
**Supervisor:** Dr. M. Gangappa (Associate Professor)  
**Department:** Computer Science and Engineering (AIML, IoT and R&AI)  
**Institution:** VNR Vignana Jyothi Institute of Engineering and Technology (VNRVJIET)  
**Degree:** M.Tech (AIDS) — Year I, Semester II  

---

## 1. Project Overview & Elevator Pitch (30-Second Summary)

> *"Pancreatic tumor detection from abdominal CT scans is one of the hardest medical imaging challenges because pancreatic tumors are small, have low soft-tissue contrast, irregular boundaries, and complex surrounding vascular anatomy. Standard 3D CNNs fail because local convolutions cannot model long-range anatomical context.*
>
> *In this project, I developed **PancreasGATUNet**—a novel deep learning framework combining a **3D CNN Encoder-Decoder** with a **GATv2 Graph Attention Network** in the bottleneck. This allows the network to model spatial relationships between anatomical regions, improving tumor segmentation Dice score to **78.40%** (up from 68.20% in baseline 3D U-Net) and reducing 3D bounding box centroid localization error to **14.2 mm**."*

---

## 2. Key Slide-by-Slide Presentation Breakdown

### Slide 1: Introduction & Problem Statement
- **Challenge 1 (Small Target Size):** Pancreatic tumors occupy $< 1\%$ of the total abdominal CT volumetric space.
- **Challenge 2 (Low Soft-Tissue Contrast):** CT attenuation values (Hounsfield Units) of tumor tissue overlap heavily with healthy pancreatic tissue.
- **Challenge 3 (Convolution Limitation):** Standard 3D CNN filters have a small local receptive field and miss long-range context between distant anatomical structures.

### Slide 2: Objectives of the Work
1. **Multi-Class Volumetric Segmentation:** Simultaneously segment the **Background (Class 0)**, **Pancreas Organ (Class 1)**, and **Pancreatic Tumor (Class 2)**.
2. **Automated 3D Bounding-Box Localization:** Extract precise 3D spatial coordinate boundaries $(X_{\min}, X_{\max}, Y_{\min}, Y_{\max}, Z_{\min}, Z_{\max})$ and centroid locations for targeted radiologist review.
3. **Graph Attention Integration:** Incorporate GATv2 graph self-attention over spatial sub-volume nodes to disambiguate soft-tissue boundaries.

### Slide 3: Dataset & Medical Preprocessing
- **Dataset:** **Medical Segmentation Decathlon (MSD) Task07 Pancreas**
  - 281 3D abdominal CT scans with expert voxel-level ground truth annotations.
- **Preprocessing Pipeline:**
  1. **Hounsfield Unit (HU) Windowing:** Clipped CT attenuation values to `[-100, 240] HU` to suppress uninformative bone/air voxels and highlight soft abdominal organs.
  2. **Isotropic Resampling:** Resampled all scans to a uniform voxel spacing of $1.0 \times 1.0 \times 1.0 \text{ mm}^3$.
  3. **Z-Score Intensity Normalization:** Normalized CT voxel intensities to zero mean and unit variance.
  4. **Foreground/Background Patch Sampling:** Extracted balanced $96 \times 96 \times 96$ sub-volume patches containing at least $50\%$ organ/tumor voxels.

### Slide 4: Proposed Architecture — `PancreasGATUNet`
1. **3D CNN Encoder:** 4 resolution levels of 3D volumetric convolutions extracting multi-scale feature maps $(16 \rightarrow 32 \rightarrow 64 \rightarrow 128$ channels).
2. **3D Spatial Graph Construction:**
   - Bottleneck features are divided into a 3D spatial grid of nodes ($4 \times 4 \times 4 = 64$ graph nodes).
   - Constructed a $k$-Nearest Neighbors ($k=8$) spatial graph based on Euclidean node coordinates.
3. **GATv2 Multi-Head Graph Attention Module:**
   - Applies multi-head self-attention ($\text{Heads} = 2, \text{Layers} = 1$) across anatomical nodes.
   - Computes dynamic attention weight $\alpha_{ij}$ between node $i$ and node $j$:
     $$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W} [\mathbf{h}_i \,||\, \mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W} [\mathbf{h}_i \,||\, \mathbf{h}_k]\right)\right)}$$
4. **Graph-to-Volume Feature Fusion:** Projects updated graph embeddings back into 3D spatial feature tensors and merges them via skip connections.
5. **Dual-Head 3D Decoder:**
   - **Head A (Segmentation):** Predicts 3D voxel probability maps for 3 classes.
   - **Head B (Localization):** Predicts 3D heatmap centroid distributions and regresses bounding box coordinates.

### Slide 5: Composite Multi-Task Loss Function
To address extreme foreground/background class imbalance, the model is trained with a composite loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \lambda_1 \mathcal{L}_{\text{Focal}} + \lambda_2 \mathcal{L}_{\text{BBox}}$$

- **Dice Loss ($\mathcal{L}_{\text{Dice}}$):** Maximizes volumetric spatial overlap.
- **Focal Loss ($\mathcal{L}_{\text{Focal}}$ with $\gamma=2.0$):** Suppresses easy background voxels and focuses gradient updates on difficult tumor boundary voxels.
- **Smooth L1 Loss ($\mathcal{L}_{\text{BBox}}$):** Penalizes 3D bounding box coordinate localization errors.

### Slide 6: Quantitative Results & Metrics
- **Pancreas (Organ):**
  - Dice Score: **86.82%**
  - Precision: **76.81%** | Recall: **99.83%** | IoU: **76.71%** | Accuracy: **94.77%**
- **Pancreatic Tumor:**
  - Dice Score: **78.40%**
  - Precision: **81.20%** | Recall: **76.10%** | IoU: **65.10%** | Accuracy: **94.80%**
  - 95% Hausdorff Distance (HD95): **12.40 mm** (down from 22.8 mm in baseline)
- **3D Bounding Box Localization:**
  - Centroid Distance Error: **14.20 mm**
  - Bounding Box IoU: **0.7100**

### Slide 7: Baseline Comparison & Discussion
- **Standard 3D U-Net:** Tumor Dice = $68.20\%$, HD95 = $22.80 \text{ mm}$
- **3D Attention U-Net:** Tumor Dice = $72.90\%$, HD95 = $17.50 \text{ mm}$
- **Proposed PancreasGATUNet (Ours):** Tumor Dice = **78.40%**, HD95 = **12.40 mm**

*Conclusion:* Graph attention provides a **$+10.2\%$ improvement in Tumor Dice** over conventional 3D U-Net by capturing contextual organ relationships.

---

## 3. Top Viva & Panel Q&A Preparation

### Q1: Why did you use HU Windowing between -100 and +240?
**Answer:** Raw CT scans have intensity values ranging from $-1000 \text{ HU}$ (air) to $+3000 \text{ HU}$ (dense bone). Soft tissue organs like the pancreas and tumors lie in the narrow range of $0 \text{ to } +150 \text{ HU}$. By clipping intensities to `[-100, +240] HU`, we remove uninformative background noise and amplify contrast around soft abdominal organs.

### Q2: Why GATv2 instead of standard GAT or Transformer Self-Attention?
**Answer:** Standard GAT uses a static attention mechanism where the ranking of attention weights is independent of the query node. GATv2 fixes this by placing the non-linearity before the inner product, making attention strictly dynamic and query-dependent. Compared to 3D Transformers, GATv2 is far more computationally lightweight because it operates on sparse $k$-NN spatial graph nodes ($k=8$) rather than dense voxel pairs.

### Q3: How do you extract 3D Bounding Boxes from segmentation masks?
**Answer:** Once the network outputs the 3D tumor probability map, we apply connected-component analysis and thresholding to find the contiguous positive voxel cluster. We calculate the minimum and maximum indices along each spatial dimension $(X_{\min}, X_{\max}, Y_{\min}, Y_{\max}, Z_{\min}, Z_{\max})$ to bound the tumor volume, and compute its 3D spatial centroid.

### Q4: Why is Dice Score for Tumor lower than Pancreas organ (78.4% vs 86.8%)?
**Answer:** Pancreatic tumors are much smaller in volume than the organ itself, leading to higher sensitivity to boundary voxel misclassifications. Additionally, tumor tissue seamlessly infiltrates healthy organ tissue, creating vague, low-contrast boundary gradients.

### Q5: What is the 95th Percentile Hausdorff Distance (HD95)?
**Answer:** HD95 measures the spatial distance between the boundary points of the predicted segmentation mask and the ground truth mask. The 95th percentile is used to eliminate extreme outlier noise points. A lower HD95 (12.4 mm in our method vs 22.8 mm in baseline) indicates tighter, clinically reliable boundary agreement.

---

## 4. Key Project Files Checklist for Demo
1. `train.py`: Main model training script with AMP (Automatic Mixed Precision).
2. `evaluation/evaluate.py`: Generates all quantitative metrics and JSON reports.
3. `visualization/generate_presentation_bar_charts.py`: Generates 300-DPI presentation bar graphs.
4. `visualization/visualize_results.py`: Renders multi-planar CT slice overlays and 3D BBox overlays.
5. `models/full_gat_unet.py`: Core architecture implementing 3D UNet + GATv2.
