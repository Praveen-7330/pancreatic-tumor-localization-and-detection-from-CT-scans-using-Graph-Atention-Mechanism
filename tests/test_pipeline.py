import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from models.full_gat_unet import PancreasGATUNet
from training.losses import CompositePancreasLoss
from training.metrics import evaluate_pancreas_and_tumor
from evaluation.postprocessing import extract_3d_bounding_box

def run_tests():
    print("[*] Running Integration Tests...")
    
    # 1. Forward Pass Test
    B, C, D, H, W = 2, 1, 64, 64, 64
    dummy_ct = torch.randn(B, C, D, H, W)
    model = PancreasGATUNet(
        in_channels=1, num_classes=3, feature_channels=(16, 32, 64, 128),
        bottleneck_dim=128, grid_size=(4, 4, 4), gat_hidden_dim=64, gat_heads=2, gat_layers=1, k_neighbors=4
    )
    seg_logits, loc_heatmap, attn = model(dummy_ct)
    assert seg_logits.shape == (B, 3, D, H, W), f"Wrong shape: {seg_logits.shape}"
    print("[PASS] 3D CNN + GAT Model Forward Pass Test")
    
    # 2. Loss and Backpropagation Test
    criterion = CompositePancreasLoss(num_classes=3)
    targets = torch.randint(0, 3, (B, 1, D, H, W))
    loss, loss_dict = criterion(seg_logits, loc_heatmap, targets)
    loss.backward()
    assert loss.item() > 0
    print("[PASS] Composite Loss and Backpropagation Test")
    
    # 3. Evaluation and Bounding Box Test
    gt = np.zeros((30, 30, 30), dtype=np.int64)
    pred = np.zeros((30, 30, 30), dtype=np.int64)
    gt[5:15, 5:15, 5:15] = 2
    pred[5:15, 5:15, 5:15] = 2
    metrics = evaluate_pancreas_and_tumor(pred, gt)
    assert np.isclose(metrics["tumor_dice"], 1.0)
    bbox, _ = extract_3d_bounding_box(pred == 2)
    assert bbox["z_min"] == 5 and bbox["z_max"] == 14
    print("[PASS] Metrics and 3D Bounding Box Extraction Test")
    
    print("\n========================================")
    print("[ALL UNIT AND INTEGRATION TESTS PASSED]")
    print("========================================\n")

if __name__ == "__main__":
    run_tests()
