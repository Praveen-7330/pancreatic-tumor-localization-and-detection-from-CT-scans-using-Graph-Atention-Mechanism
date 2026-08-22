import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

from data.synthetic_generator import generate_synthetic_ct_volume
from models.full_gat_unet import PancreasGATUNet
from evaluation.postprocessing import extract_3d_bounding_box
from evaluation.sliding_window_infer import predict_full_volume

def plot_multi_planar_comparison(ct_vol, gt_vol, pred_vol, save_path="outputs/visualizations/multi_planar_results.png"):
    """
    Renders Axial, Coronal, and Sagittal slice overlays of CT scan with Pancreas (Green) and Tumor (Red).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Locate slice indices with maximum tumor volume or center of volume
    pos = np.where(gt_vol == 2)
    if len(pos[0]) > 0:
        z_slice = int(np.mean(pos[0]))
        y_slice = int(np.mean(pos[1]))
        x_slice = int(np.mean(pos[2]))
    else:
        D, H, W = ct_vol.shape
        z_slice, y_slice, x_slice = D // 2, H // 2, W // 2
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("PancreasGATUNet: Ground Truth vs Model Prediction", fontsize=16, fontweight="bold")
    
    # Colormap: Background=Transparent, Pancreas=Light Green, Tumor=Red
    cmap_mask = ListedColormap(["none", "#2ecc71", "#e74c3c"])
    
    slices_gt = [ct_vol[z_slice, :, :], ct_vol[:, y_slice, :], ct_vol[:, :, x_slice]]
    masks_gt = [gt_vol[z_slice, :, :], gt_vol[:, y_slice, :], gt_vol[:, :, x_slice]]
    
    slices_pred = [ct_vol[z_slice, :, :], ct_vol[:, y_slice, :], ct_vol[:, :, x_slice]]
    masks_pred = [pred_vol[z_slice, :, :], pred_vol[:, y_slice, :], pred_vol[:, :, x_slice]]
    
    titles = ["Axial View (Z-plane)", "Coronal View (Y-plane)", "Sagittal View (X-plane)"]
    
    for j in range(3):
        # Ground Truth Row
        axes[0, j].imshow(slices_gt[j], cmap="gray", origin="lower")
        axes[0, j].imshow(masks_gt[j], cmap=cmap_mask, alpha=0.5, origin="lower", vmin=0, vmax=2)
        axes[0, j].set_title(f"GT: {titles[j]}", fontsize=12, fontweight="bold")
        axes[0, j].axis("off")
        
        # Prediction Row
        axes[1, j].imshow(slices_pred[j], cmap="gray", origin="lower")
        axes[1, j].imshow(masks_pred[j], cmap=cmap_mask, alpha=0.5, origin="lower", vmin=0, vmax=2)
        axes[1, j].set_title(f"Pred: {titles[j]}", fontsize=12, fontweight="bold")
        axes[1, j].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved multi-planar CT overlay figure -> {save_path}")

def plot_3d_bounding_box_overlay(ct_vol, gt_vol, pred_vol, save_path="outputs/visualizations/3d_bbox_localization.png"):
    """
    Renders 3D Bounding Box bounding rectangle overlay on Axial CT slice.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gt_bbox, gt_cent = extract_3d_bounding_box(gt_vol == 2)
    pred_bbox, pred_cent = extract_3d_bounding_box(pred_vol == 2)
    
    z_slice = gt_bbox["z_min"] + (gt_bbox["z_max"] - gt_bbox["z_min"]) // 2 if gt_bbox else ct_vol.shape[0] // 2
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(ct_vol[z_slice, :, :], cmap="gray", origin="lower")
    
    if gt_bbox:
        rect_gt = Rectangle(
            (gt_bbox["x_min"], gt_bbox["y_min"]),
            gt_bbox["x_max"] - gt_bbox["x_min"],
            gt_bbox["y_max"] - gt_bbox["y_min"],
            linewidth=2.5, edgecolor="#2ecc71", facecolor="none", label="GT Tumor BBox"
        )
        ax.add_patch(rect_gt)
        ax.plot(gt_cent[2], gt_cent[1], "go", markersize=8, label="GT Centroid")
        
    if pred_bbox:
        rect_pred = Rectangle(
            (pred_bbox["x_min"], pred_bbox["y_min"]),
            pred_bbox["x_max"] - pred_bbox["x_min"],
            pred_bbox["y_max"] - pred_bbox["y_min"],
            linewidth=2.5, edgecolor="#e74c3c", linestyle="--", facecolor="none", label="Predicted Tumor BBox"
        )
        ax.add_patch(rect_pred)
        ax.plot(pred_cent[2], pred_cent[1], "rx", markersize=10, markeredgewidth=2.5, label="Pred Centroid")
        
    ax.set_title("3D Tumor Bounding Box & Centroid Localization", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.axis("off")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved 3D Bounding Box overlay figure -> {save_path}")

def plot_gat_attention_heatmap(attn_weights, save_path="outputs/visualizations/gat_attention_matrix.png"):
    """
    Plots GAT multi-head relational attention weight matrix across volumetric node grid.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if attn_weights is None:
        print("[!] Attention weights tensor is None.")
        return
        
    # attn_weights shape: (B, num_heads, N, N)
    attn_np = attn_weights[0].mean(dim=0).cpu().numpy()  # Average across attention heads -> (N, N)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(attn_np, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_title("GATv2 Volumetric Node-to-Node Attention Weights", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Volumetric Node Index")
    ax.set_ylabel("Source Volumetric Node Index")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved GAT attention matrix heatmap figure -> {save_path}")

def main():
    print("[*] Generating PancreasGATUNet Master's Project Visualizations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate sample synthetic CT volume
    ct_vol, gt_vol = generate_synthetic_ct_volume(spatial_shape=(96, 96, 96), seed=100)
    
    # 2. Run Forward Pass with PancreasGATUNet
    model = PancreasGATUNet(
        in_channels=1, num_classes=3, feature_channels=(16, 32, 64, 128),
        bottleneck_dim=128, grid_size=(4, 4, 4), gat_hidden_dim=64, gat_heads=2, gat_layers=1, k_neighbors=4
    ).to(device)
    model.eval()
    
    ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_logits, loc_heatmap, attn_weights = model(ct_tensor)
        pred_vol = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        
    # 3. Render Figures
    plot_multi_planar_comparison(ct_vol, gt_vol, pred_vol, save_path="outputs/visualizations/multi_planar_results.png")
    plot_3d_bounding_box_overlay(ct_vol, gt_vol, pred_vol, save_path="outputs/visualizations/3d_bbox_localization.png")
    plot_gat_attention_heatmap(attn_weights, save_path="outputs/visualizations/gat_attention_matrix.png")
    print("[*] All visualization figures generated successfully in 'outputs/visualizations/'.")

if __name__ == "__main__":
    main()
