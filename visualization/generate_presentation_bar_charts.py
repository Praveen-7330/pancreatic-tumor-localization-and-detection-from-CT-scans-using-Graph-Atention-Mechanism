import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output directory for presentation charts
output_dir = os.path.join("outputs", "presentation_graphs")
os.makedirs(output_dir, exist_ok=True)

# ── Styling Config (Modern Dark Presentation Theme) ──────────────────────────
BG_COLOR = "#0F172A"       # Slate 900
PANEL_COLOR = "#1E293B"    # Slate 800
TEXT_COLOR = "#F8FAFC"     # Slate 50
SUBTEXT_COLOR = "#94A3B8"  # Slate 400
GRID_COLOR = "#334155"     # Slate 700

COLOR_PANCREAS = "#38BDF8"  # Sky Blue
COLOR_TUMOR = "#F43F5E"     # Rose Red
COLOR_UNET = "#64748B"      # Slate Gray
COLOR_ATTN_UNET = "#F59E0B" # Amber Yellow
COLOR_OURS = "#10B981"      # Emerald Green

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

def apply_theme(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.2)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)

# -----------------------------------------------------------------------------
# GRAPH 1: Proposed Method — Pancreas vs Tumor Across All Key Metrics
# -----------------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Dice (DSC)", "IoU (Jaccard)"]
pancreas_scores = [94.77, 76.81, 99.83, 86.82, 86.82, 76.71]
tumor_scores    = [94.80, 81.20, 76.10, 78.40, 78.40, 65.10]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
rects1 = ax.bar(x - width/2, pancreas_scores, width, label="Pancreas Organ", color=COLOR_PANCREAS, edgecolor=BG_COLOR, linewidth=1.5, alpha=0.95)
rects2 = ax.bar(x + width/2, tumor_scores, width, label="Pancreatic Tumor", color=COLOR_TUMOR, edgecolor=BG_COLOR, linewidth=1.5, alpha=0.95)

ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
ax.set_title("PancreasGATUNet — Pancreas Organ vs. Tumor Segmentation Performance", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=12, loc="upper right")

# Add data values on top of bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=9.5, fontweight="bold")

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=9.5, fontweight="bold")

apply_theme(fig, ax)
fig.tight_layout()
p1 = os.path.join(output_dir, "presentation_bar_chart_metrics.png")
fig.savefig(p1, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"[+] Saved Graph 1: {p1}")

# -----------------------------------------------------------------------------
# GRAPH 2: Architectural Comparison — Baseline 3D UNet vs Attention UNet vs Proposed PancreasGATUNet
# -----------------------------------------------------------------------------
models = ["Standard 3D U-Net", "3D Attention U-Net", "Proposed PancreasGATUNet\n(Ours with GATv2)"]
tumor_dice_scores = [68.20, 72.90, 78.40]
pancreas_dice_scores = [81.40, 84.10, 86.82]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
rects1 = ax.bar(x - width/2, pancreas_dice_scores, width, label="Pancreas Dice Score (%)", color="#38BDF8", edgecolor=BG_COLOR, linewidth=1.5)
rects2 = ax.bar(x + width/2, tumor_dice_scores, width, label="Tumor Dice Score (%)", color="#F43F5E", edgecolor=BG_COLOR, linewidth=1.5)

ax.set_ylabel("Dice Similarity Coefficient (%)", fontsize=13, fontweight="bold")
ax.set_title("Architectural Baseline Benchmark — Dice Score Comparison", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=12, loc="lower right")

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f"{height:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f"{height:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

apply_theme(fig, ax)
fig.tight_layout()
p2 = os.path.join(output_dir, "presentation_bar_chart_comparison.png")
fig.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"[+] Saved Graph 2: {p2}")

# -----------------------------------------------------------------------------
# GRAPH 3: Boundary & Distance Error Comparison (HD95 & Centroid Error - Lower is Better)
# -----------------------------------------------------------------------------
models_dist = ["Standard 3D U-Net", "3D Attention U-Net", "Proposed PancreasGATUNet\n(Ours)"]
hd95_vals = [22.80, 17.50, 12.40]       # mm
centroid_vals = [28.40, 19.80, 14.20]   # mm

x = np.arange(len(models_dist))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
rects1 = ax.bar(x - width/2, hd95_vals, width, label="Tumor 95% Hausdorff Distance (HD95 in mm)", color="#A855F7", edgecolor=BG_COLOR, linewidth=1.5)
rects2 = ax.bar(x + width/2, centroid_vals, width, label="3D Bounding Box Centroid Error (mm)", color="#EAB308", edgecolor=BG_COLOR, linewidth=1.5)

ax.set_ylabel("Distance Error (mm) — Lower is Better", fontsize=13, fontweight="bold")
ax.set_title("Boundary Distance & Centroid Localization Error (mm)", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models_dist, fontsize=11, fontweight="bold")
ax.set_ylim(0, 35)
ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=11, loc="upper right")

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f"{height:.1f} mm",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f"{height:.1f} mm",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")

apply_theme(fig, ax)
fig.tight_layout()
p3 = os.path.join(output_dir, "presentation_bar_chart_distance_errors.png")
fig.savefig(p3, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"[+] Saved Graph 3: {p3}")

print("\n[SUCCESS] All presentation bar graphs generated successfully!")
