"""
========================================================================================
PancreasGATUNet: Master's Mini Project Interactive Demonstration Script
Author: Kanaparthi Praveen (Roll No: 25071DB205)
Supervisor: Dr. M. Gangappa
========================================================================================
"""

import sys
from pathlib import Path
# Ensure project root is in path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
import numpy as np
from data.synthetic_generator import generate_synthetic_ct_volume
from models.full_gat_unet import PancreasGATUNet
from evaluation.postprocessing import extract_3d_bounding_box
from evaluation.sliding_window_infer import predict_full_volume
from visualization.visualize_results import plot_multi_planar_comparison, plot_3d_bounding_box_overlay, plot_gat_attention_heatmap

def run_demo():
    print("========================================================================")
    print("  PancreasGATUNet Master's Mini Project Interactive Demo")
    print("========================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    # 1. Generate Synthetic 3D CT Volume
    print("\n[Step 1] Simulating 3D Abdominal CT Volume (96x96x96 voxels)...")
    ct_vol, gt_vol = generate_synthetic_ct_volume(spatial_shape=(96, 96, 96), seed=2026)
    print(f"  [+] CT Volume HU Range: [{ct_vol.min():.1f}, {ct_vol.max():.1f}]")
    print(f"  [+] Ground Truth Voxels -> Background: {(gt_vol==0).sum()}, Pancreas: {(gt_vol==1).sum()}, Tumor: {(gt_vol==2).sum()}")
    
    # 2. Instantiate Model
    print("\n[Step 2] Initializing 3D Graph Attention UNet (PancreasGATUNet)...")
    model = PancreasGATUNet(
        in_channels=1,
        num_classes=3,
        feature_channels=(16, 32, 64, 128),
        bottleneck_dim=128,
        grid_size=(4, 4, 4),
        gat_hidden_dim=64,
        gat_heads=4,
        gat_layers=2,
        k_neighbors=4
    ).to(device)
    model.eval()
    print("  [+] Model Parameters Count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # 3. Model Forward Pass & Attention Extraction
    print("\n[Step 3] Executing Model Forward Pass & Extracting GAT Graph Attention Weights...")
    ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_logits, loc_heatmap, attn_weights = model(ct_tensor)
        pred_vol = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        
    print(f"  [+] Output Segmentation Logits Shape: {seg_logits.shape}")
    print(f"  [+] Output Localization Heatmap Shape: {loc_heatmap.shape}")
    print(f"  [+] Volumetric Attention Matrix Shape: {attn_weights.shape}")
    
    # 4. 3D Bounding Box Extraction
    print("\n[Step 4] Extracting 3D Tumor Bounding Box Coordinates & Centroid...")
    gt_bbox, gt_cent = extract_3d_bounding_box(gt_vol == 2)
    pred_bbox, pred_cent = extract_3d_bounding_box(pred_vol == 2)
    
    if gt_bbox:
        print(f"  [GT Tumor BBox] Z:[{gt_bbox['z_min']}-{gt_bbox['z_max']}], Y:[{gt_bbox['y_min']}-{gt_bbox['y_max']}], X:[{gt_bbox['x_min']}-{gt_bbox['x_max']}] | Volume: {gt_bbox['volume_voxels']} voxels")
    if pred_bbox:
        print(f"  [Pred Tumor BBox] Z:[{pred_bbox['z_min']}-{pred_bbox['z_max']}], Y:[{pred_bbox['y_min']}-{pred_bbox['y_max']}], X:[{pred_bbox['x_min']}-{pred_bbox['x_max']}] | Volume: {pred_bbox['volume_voxels']} voxels")
        
    # 5. Generate Figures
    print("\n[Step 5] Rendering Visualization Figures...")
    plot_multi_planar_comparison(ct_vol, gt_vol, pred_vol, save_path="outputs/visualizations/demo_multi_planar.png")
    plot_3d_bounding_box_overlay(ct_vol, gt_vol, pred_vol, save_path="outputs/visualizations/demo_3d_bbox.png")
    plot_gat_attention_heatmap(attn_weights, save_path="outputs/visualizations/demo_gat_heatmap.png")
    
    print("\n========================================================================")
    print("  DEMO COMPLETED SUCCESSFULLY! All outputs saved in 'outputs/visualizations/'")
    print("========================================================================")

if __name__ == "__main__":
    run_demo()
