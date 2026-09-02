import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import yaml
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files without needing a display
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff

from data.dataset import load_msd_pancreas_files, get_val_only_loader
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

def evaluate_model(model, val_loader, device, config, max_cases=None):
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_cases and i >= max_cases:
                print(f"[INFO] Stopping early after {max_cases} cases (--max_cases).")
                break
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
                "case_index":            i + 1,
                # --- Pancreas ---
                "pancreas_dice":          panc_m["dice"],
                "pancreas_iou":           panc_m["iou"],
                "pancreas_precision":     panc_m["precision"],
                "pancreas_recall":        panc_m["recall"],
                "pancreas_f1":            panc_m["f1"],
                "pancreas_accuracy":      panc_m["accuracy"],
                "pancreas_hd95_mm":       panc_hd95,
                # --- Tumor ---
                "tumor_dice":             tumor_m["dice"],
                "tumor_iou":              tumor_m["iou"],
                "tumor_precision":        tumor_m["precision"],
                "tumor_recall":           tumor_m["recall"],
                "tumor_f1":               tumor_m["f1"],
                "tumor_accuracy":         tumor_m["accuracy"],
                "tumor_hd95_mm":          tumor_hd95,
                # --- BBox ---
                "bbox_centroid_dist_mm": bbox_err["centroid_distance"],
                "bbox_iou":              bbox_err["bbox_iou"],
            }
            results.append(case_res)
            print(
                f"[Case {i+1:02d}] "
                f"Panc  — Dice: {panc_m['dice']:.4f}  Prec: {panc_m['precision']:.4f}  Rec: {panc_m['recall']:.4f}  F1: {panc_m['f1']:.4f}  Acc: {panc_m['accuracy']:.4f} | "
                f"Tumor — Dice: {tumor_m['dice']:.4f}  Prec: {tumor_m['precision']:.4f}  Rec: {tumor_m['recall']:.4f}  F1: {tumor_m['f1']:.4f}  Acc: {tumor_m['accuracy']:.4f} | "
                f"HD95: {tumor_hd95:.2f}mm  BBox: {bbox_err['centroid_distance']:.2f}mm"
            )
            
    # ── Summary Statistics ──────────────────────────────────────────────
    def _mean(key):  return float(np.mean([r[key] for r in results]))
    def _std(key):   return float(np.std( [r[key] for r in results]))

    tumor_hd95s = [r["tumor_hd95_mm"]          for r in results if r["tumor_hd95_mm"]          < 900]
    bbox_dists  = [r["bbox_centroid_dist_mm"]   for r in results if r["bbox_centroid_dist_mm"]  < 900]

    summary = {
        "num_cases": len(results),
        # Pancreas
        "mean_pancreas_dice":      _mean("pancreas_dice"),
        "std_pancreas_dice":       _std( "pancreas_dice"),
        "mean_pancreas_precision": _mean("pancreas_precision"),
        "mean_pancreas_recall":    _mean("pancreas_recall"),
        "mean_pancreas_f1":        _mean("pancreas_f1"),
        "mean_pancreas_accuracy":  _mean("pancreas_accuracy"),
        # Tumor
        "mean_tumor_dice":         _mean("tumor_dice"),
        "std_tumor_dice":          _std( "tumor_dice"),
        "mean_tumor_precision":    _mean("tumor_precision"),
        "mean_tumor_recall":       _mean("tumor_recall"),
        "mean_tumor_f1":           _mean("tumor_f1"),
        "mean_tumor_accuracy":     _mean("tumor_accuracy"),
        # Distance / BBox
        "mean_tumor_hd95_mm":          float(np.mean(tumor_hd95s)) if tumor_hd95s else 0.0,
        "mean_bbox_centroid_dist_mm":   float(np.mean(bbox_dists))  if bbox_dists  else 0.0,
        "cases": results,
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate PancreasGATUNet on CT Test Set")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for evaluation")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit evaluation to N cases (for quick demos)")
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
    val_loader = get_val_only_loader(
        data_dicts, fold=0, n_splits=5, num_workers=0
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
        
    summary = evaluate_model(model, val_loader, device, config, max_cases=args.max_cases)
    
    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, "evaluation_results.json")

    # Annotate JSON with project/model context
    summary["project"]    = "Pancreatic Tumor Localization & Segmentation"
    summary["model"]      = "PancreasGATUNet (UNet + Graph Attention Network)"
    summary["dataset"]    = "Medical Segmentation Decathlon — Task07 Pancreas (MSD)"
    summary["checkpoint"] = args.checkpoint

    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  PancreasGATUNet — Evaluation Results")
    print(f"  Model  : UNet + Graph Attention Network (GAT)")
    print(f"  Dataset: MSD Task07 Pancreas  |  Cases: {summary['num_cases']}")
    print(sep)
    print(f"  {'Metric':<32} {'Pancreas':>10}  {'Tumor':>10}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}")
    print(f"  {'Accuracy':<32} {summary['mean_pancreas_accuracy']:>10.4f}  {summary['mean_tumor_accuracy']:>10.4f}")
    print(f"  {'Precision':<32} {summary['mean_pancreas_precision']:>10.4f}  {summary['mean_tumor_precision']:>10.4f}")
    print(f"  {'Recall':<32} {summary['mean_pancreas_recall']:>10.4f}  {summary['mean_tumor_recall']:>10.4f}")
    print(f"  {'F1 Score':<32} {summary['mean_pancreas_f1']:>10.4f}  {summary['mean_tumor_f1']:>10.4f}")
    print(f"  {'Dice':<32} {summary['mean_pancreas_dice']:>10.4f}  {summary['mean_tumor_dice']:>10.4f}")
    print(f"  {'Dice Std':<32} {summary['std_pancreas_dice']:>10.4f}  {summary['std_tumor_dice']:>10.4f}")
    print(sep)
    print(f"  Mean Tumor HD95          : {summary['mean_tumor_hd95_mm']:.2f} mm")
    print(f"  Mean BBox Centroid Error : {summary['mean_bbox_centroid_dist_mm']:.2f} mm")
    print(sep)
    print(f"[*] Saved evaluation report to: {res_path}")

    plot_evaluation_results(summary, args.output_dir)

def plot_evaluation_results(summary, output_dir):
    """Generate and save evaluation metric graphs to output_dir."""
    import matplotlib.pyplot as plt
    cases     = summary["cases"]
    case_ids  = [c["case_index"] for c in cases]
    os.makedirs(output_dir, exist_ok=True)

    # ── Colour palette ────────────────────────────────────────────────────
    PANC_COLOR  = "#4C9BE8"
    TUMOR_COLOR = "#E8754C"
    BG          = "#1A1A2E"
    GRID        = "#2E2E4E"

    def _style(fig, ax_list):
        fig.patch.set_facecolor(BG)
        for ax in ax_list:
            ax.set_facecolor(BG)
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.spines[:].set_color(GRID)
            ax.grid(color=GRID, linestyle="--", linewidth=0.6)

    # ── Graph 1: Grouped bar chart — mean metrics ─────────────────────────
    metrics      = ["Accuracy", "Precision", "Recall", "F1 Score", "Dice"]
    panc_vals    = [
        summary["mean_pancreas_accuracy"],
        summary["mean_pancreas_precision"],
        summary["mean_pancreas_recall"],
        summary["mean_pancreas_f1"],
        summary["mean_pancreas_dice"],
    ]
    tumor_vals   = [
        summary["mean_tumor_accuracy"],
        summary["mean_tumor_precision"],
        summary["mean_tumor_recall"],
        summary["mean_tumor_f1"],
        summary["mean_tumor_dice"],
    ]
    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, panc_vals,  w, label="Pancreas", color=PANC_COLOR,  alpha=0.9)
    ax.bar(x + w/2, tumor_vals, w, label="Tumor",    color=TUMOR_COLOR, alpha=0.9)
    for i, (pv, tv) in enumerate(zip(panc_vals, tumor_vals)):
        ax.text(i - w/2, pv + 0.01, f"{pv:.3f}", ha="center", va="bottom", color="white", fontsize=8)
        ax.text(i + w/2, tv + 0.01, f"{tv:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("PancreasGATUNet — Mean Evaluation Metrics (Pancreas vs Tumor)")
    ax.legend(facecolor=GRID, labelcolor="white")
    _style(fig, [ax])
    p1 = os.path.join(output_dir, "graph1_mean_metrics.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[*] Saved: {p1}")

    # ── Graph 2: Per-case Dice & F1 line chart ────────────────────────────
    panc_dice  = [c["pancreas_dice"]  for c in cases]
    tumor_dice = [c["tumor_dice"]     for c in cases]
    panc_f1    = [c["pancreas_f1"]    for c in cases]
    tumor_f1   = [c["tumor_f1"]       for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(case_ids, panc_dice,  marker="o", color=PANC_COLOR,  label="Pancreas Dice",  linewidth=1.5)
    axes[0].plot(case_ids, tumor_dice, marker="s", color=TUMOR_COLOR, label="Tumor Dice",     linewidth=1.5)
    axes[0].axhline(np.mean(panc_dice),  color=PANC_COLOR,  linestyle="--", linewidth=1, alpha=0.6)
    axes[0].axhline(np.mean(tumor_dice), color=TUMOR_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_title("PancreasGATUNet — Dice Score per Case")
    axes[0].set_xlabel("Case")
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(facecolor=GRID, labelcolor="white")
    axes[1].plot(case_ids, panc_f1,  marker="o", color=PANC_COLOR,  label="Pancreas F1",  linewidth=1.5)
    axes[1].plot(case_ids, tumor_f1, marker="s", color=TUMOR_COLOR, label="Tumor F1",     linewidth=1.5)
    axes[1].axhline(np.mean(panc_f1),  color=PANC_COLOR,  linestyle="--", linewidth=1, alpha=0.6)
    axes[1].axhline(np.mean(tumor_f1), color=TUMOR_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_title("PancreasGATUNet — F1 Score per Case")
    axes[1].set_xlabel("Case")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(facecolor=GRID, labelcolor="white")
    _style(fig, axes)
    p2 = os.path.join(output_dir, "graph2_per_case_dice_f1.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[*] Saved: {p2}")

    # ── Graph 3: Box plots — metric distributions ─────────────────────────
    box_data  = {
        "Panc\nAccuracy":  [c["pancreas_accuracy"]  for c in cases],
        "Panc\nPrecision": [c["pancreas_precision"]  for c in cases],
        "Panc\nRecall":    [c["pancreas_recall"]     for c in cases],
        "Panc\nF1":        [c["pancreas_f1"]         for c in cases],
        "Tumor\nAccuracy": [c["tumor_accuracy"]      for c in cases],
        "Tumor\nPrecision":[c["tumor_precision"]     for c in cases],
        "Tumor\nRecall":   [c["tumor_recall"]        for c in cases],
        "Tumor\nF1":       [c["tumor_f1"]            for c in cases],
    }
    colors = ([PANC_COLOR] * 4) + ([TUMOR_COLOR] * 4)
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(
        list(box_data.values()),
        tick_labels=list(box_data.keys()),
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("PancreasGATUNet — Metric Distribution across Cases (Box Plot)")
    _style(fig, [ax])
    p3 = os.path.join(output_dir, "graph3_boxplots.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[*] Saved: {p3}")

    # ── Graph 4: Per-case Precision, Recall, Accuracy (Tumor) ────────────
    tumor_prec = [c["tumor_precision"] for c in cases]
    tumor_rec  = [c["tumor_recall"]    for c in cases]
    tumor_acc  = [c["tumor_accuracy"]  for c in cases]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(case_ids, tumor_prec, marker="^", color="#F0A500", label="Precision", linewidth=1.5)
    ax.plot(case_ids, tumor_rec,  marker="v", color="#00C49A", label="Recall",    linewidth=1.5)
    ax.plot(case_ids, tumor_acc,  marker="D", color="#C77DFF", label="Accuracy",  linewidth=1.5)
    ax.set_title("PancreasGATUNet — Tumor Precision, Recall & Accuracy per Case")
    ax.set_xlabel("Case")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=GRID, labelcolor="white")
    _style(fig, [ax])
    p4 = os.path.join(output_dir, "graph4_tumor_prec_rec_acc.png")
    fig.tight_layout()
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"[*] Saved: {p4}")

    print(f"\n[*] All graphs saved to: {os.path.abspath(output_dir)}/")

if __name__ == "__main__":
    main()
