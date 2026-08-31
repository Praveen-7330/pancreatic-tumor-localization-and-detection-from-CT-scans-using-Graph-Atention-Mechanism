import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff

from data.dataset import load_msd_pancreas_files, get_dataloaders
from data.synthetic_generator import save_synthetic_dataset
from models.full_gat_unet import PancreasGATUNet
from evaluation.sliding_window_infer import predict_full_volume
from evaluation.postprocessing import extract_3d_bounding_box, filter_small_connected_components
from training.metrics import compute_binary_metrics

def compute_hd95_3d(pred_mask, gt_mask, voxel_spacing=(1.0, 1.0, 1.0), max_samples=1000):
    """Computes 95th percentile Hausdorff Distance for 3D binary masks using sampled boundary points."""
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 999.0  # Max penalty if no prediction/ground truth
        
    pred_pts = np.argwhere(pred_mask) * np.array(voxel_spacing)
    gt_pts = np.argwhere(gt_mask) * np.array(voxel_spacing)
    
    if len(pred_pts) > max_samples:
        idx = np.random.choice(len(pred_pts), max_samples, replace=False)
        pred_pts = pred_pts[idx]
    if len(gt_pts) > max_samples:
        idx = np.random.choice(len(gt_pts), max_samples, replace=False)
        gt_pts = gt_pts[idx]
        
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    hd95 = max(d1, d2) * 0.95
    return float(hd95)

def compute_3d_bbox_error(pred_bbox, gt_bbox):
    """Computes Centroid Distance Error and BBox IoU between 3D bounding boxes."""
    if pred_bbox is None or gt_bbox is None:
        return {"centroid_distance": 999.0, "bbox_iou": 0.0}
        
    c_pred = np.array(pred_bbox["centroid"])
    c_gt = np.array(gt_bbox["centroid"])
    centroid_dist = float(np.linalg.norm(c_pred - c_gt))
    
    # 3D Bounding Box Intersection Over Union
    z_int = max(0, min(pred_bbox["z_max"], gt_bbox["z_max"]) - max(pred_bbox["z_min"], gt_bbox["z_min"]) + 1)
    y_int = max(0, min(pred_bbox["y_max"], gt_bbox["y_max"]) - max(pred_bbox["y_min"], gt_bbox["y_min"]) + 1)
    x_int = max(0, min(pred_bbox["x_max"], gt_bbox["x_max"]) - max(pred_bbox["x_min"], gt_bbox["x_min"]) + 1)
    
    intersection = z_int * y_int * x_int
    vol_pred = (pred_bbox["z_max"] - pred_bbox["z_min"] + 1) * (pred_bbox["y_max"] - pred_bbox["y_min"] + 1) * (pred_bbox["x_max"] - pred_bbox["x_min"] + 1)
    vol_gt = (gt_bbox["z_max"] - gt_bbox["z_min"] + 1) * (gt_bbox["y_max"] - gt_bbox["y_min"] + 1) * (gt_bbox["x_max"] - gt_bbox["x_min"] + 1)
    union = vol_pred + vol_gt - intersection
    bbox_iou = float(intersection / max(1, union))
    
    return {"centroid_distance": centroid_dist, "bbox_iou": bbox_iou}

def evaluate_model(model, val_loader, device, config):
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].squeeze(0).squeeze(0).cpu().numpy()
            
            pred_labels, pred_probs = predict_full_volume(
                model, images,
                roi_size=tuple(config["preprocessing"]["roi_size"]),
                sw_batch_size=config["inference"]["sw_batch_size"],
                overlap=config["inference"]["overlap"]
            )
            
            # Post-process tumor mask
            tumor_pred_raw = (pred_labels == 2)
            tumor_pred_clean = filter_small_connected_components(
                tumor_pred_raw, min_size=config["inference"].get("min_tumor_volume_voxels", 50)
            )
            
            clean_pred_labels = pred_labels.copy()
            clean_pred_labels[pred_labels == 2] = 0
            clean_pred_labels[tumor_pred_clean == 1] = 2
            
            # Compute Binary Metrics
            panc_m = compute_binary_metrics(clean_pred_labels == 1, labels == 1)
            tumor_m = compute_binary_metrics(clean_pred_labels == 2, labels == 2)
            
            # HD95
            panc_hd95 = compute_hd95_3d(clean_pred_labels == 1, labels == 1)
            tumor_hd95 = compute_hd95_3d(clean_pred_labels == 2, labels == 2)
            
            # 3D Bounding Box Error
            pred_bbox, _ = extract_3d_bounding_box(clean_pred_labels == 2)
            gt_bbox, _ = extract_3d_bounding_box(labels == 2)
            bbox_err = compute_3d_bbox_error(pred_bbox, gt_bbox)
            
            case_res = {
                "case_index": i + 1,
                "pancreas_dice": panc_m["dice"],
                "pancreas_iou": panc_m["iou"],
                "pancreas_hd95_mm": panc_hd95,
                "tumor_dice": tumor_m["dice"],
                "tumor_iou": tumor_m["iou"],
                "tumor_precision": tumor_m["precision"],
                "tumor_recall": tumor_m["recall"],
                "tumor_hd95_mm": tumor_hd95,
                "bbox_centroid_dist_mm": bbox_err["centroid_distance"],
                "bbox_iou": bbox_err["bbox_iou"]
            }
            results.append(case_res)
            print(f"[Case {i+1:02d}] Panc Dice: {panc_m['dice']:.4f} | Tumor Dice: {tumor_m['dice']:.4f} | Tumor HD95: {tumor_hd95:.2f}mm | BBox Dist: {bbox_err['centroid_distance']:.2f}mm")
            
    # Summary Statistics
    panc_dices = [r["pancreas_dice"] for r in results]
    tumor_dices = [r["tumor_dice"] for r in results]
    tumor_hd95s = [r["tumor_hd95_mm"] for r in results if r["tumor_hd95_mm"] < 900]
    bbox_dists = [r["bbox_centroid_dist_mm"] for r in results if r["bbox_centroid_dist_mm"] < 900]
    
    summary = {
        "num_cases": len(results),
        "mean_pancreas_dice": float(np.mean(panc_dices)),
        "std_pancreas_dice": float(np.std(panc_dices)),
        "mean_tumor_dice": float(np.mean(tumor_dices)),
        "std_tumor_dice": float(np.std(tumor_dices)),
        "mean_tumor_hd95_mm": float(np.mean(tumor_hd95s)) if tumor_hd95s else 0.0,
        "mean_bbox_centroid_dist_mm": float(np.mean(bbox_dists)) if bbox_dists else 0.0,
        "cases": results
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate PancreasGATUNet on CT Test Set")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for evaluation")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    if args.synthetic or not os.path.exists(config["data"]["data_dir"]):
        data_dir = "./dataset/Task07_Pancreas_Synthetic"
        if not os.path.exists(data_dir):
            save_synthetic_dataset(output_dir=data_dir, num_samples=6)
        config["data"]["data_dir"] = data_dir
        
    data_dicts = load_msd_pancreas_files(config["data"]["data_dir"])
    _, val_loader = get_dataloaders(
        data_dicts, fold=0, n_splits=5,
        roi_size=tuple(config["preprocessing"]["roi_size"]),
        batch_size=1, num_workers=1
    )
    
    model = PancreasGATUNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        feature_channels=config["model"]["feature_channels"],
        bottleneck_dim=config["model"]["bottleneck_dim"],
        grid_size=tuple(config["model"]["graph"]["node_grid_size"]),
        gat_hidden_dim=config["model"]["graph"]["hidden_features"],
        gat_heads=config["model"]["graph"]["num_heads"],
        gat_layers=config["model"]["graph"]["num_layers"],
        k_neighbors=config["model"]["graph"]["k_neighbors"],
        dropout=config["model"]["graph"]["dropout"]
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"[*] Loading model checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"[WARNING] Checkpoint '{args.checkpoint}' not found. Evaluating initialized baseline weights.")
        
    summary = evaluate_model(model, val_loader, device, config)
    
    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n==================================================")
    print(f"  EVALUATION SUMMARY")
    print(f"==================================================")
    print(f"  Mean Pancreas Dice : {summary['mean_pancreas_dice']:.4f} ± {summary['std_pancreas_dice']:.4f}")
    print(f"  Mean Tumor Dice    : {summary['mean_tumor_dice']:.4f} ± {summary['std_tumor_dice']:.4f}")
    print(f"  Mean Tumor HD95    : {summary['mean_tumor_hd95_mm']:.2f} mm")
    print(f"  Mean BBox Error    : {summary['mean_bbox_centroid_dist_mm']:.2f} mm")
    print(f"==================================================")
    print(f"[*] Saved evaluation report to: {res_path}")

if __name__ == "__main__":
    main()
