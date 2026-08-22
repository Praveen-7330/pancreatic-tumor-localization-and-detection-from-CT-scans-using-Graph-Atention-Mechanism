import os
import json
import argparse
import numpy as np
from pathlib import Path

def generate_synthetic_ct_volume(spatial_shape=(96, 96, 96), seed=42):
    """
    Generates a synthetic 3D CT abdominal volume with simulated Pancreas organ (Label 1)
    and Pancreatic Tumor lesion (Label 2).
    
    Returns:
        ct_volume (np.ndarray): float32 CT intensities in Hounsfield Units [-100, 240]
        label_volume (np.ndarray): int64 segmentation mask (0=bg, 1=pancreas, 2=tumor)
    """
    np.random.seed(seed)
    D, H, W = spatial_shape
    
    # 1. Background abdominal tissue (~ -50 to 50 HU with noise)
    ct_volume = np.random.normal(loc=10.0, scale=30.0, size=spatial_shape).astype(np.float32)
    label_volume = np.zeros(spatial_shape, dtype=np.int64)
    
    # Grid coordinates centered at (0, 0, 0)
    z = np.linspace(-1, 1, D)
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")
    
    # 2. Pancreas Organ (Label 1): Curved ellipsoid around center
    cz_p, cy_p, cx_p = 0.0, 0.1, -0.1
    pancreas_mask = (
        ((grid_z - cz_p) / 0.35) ** 2 +
        ((grid_y - cy_p) / 0.25) ** 2 +
        ((grid_x - cx_p) / 0.50) ** 2
    ) <= 1.0
    
    label_volume[pancreas_mask] = 1
    ct_volume[pancreas_mask] = np.random.normal(loc=60.0, scale=15.0, size=pancreas_mask.sum()).astype(np.float32)
    
    # 3. Pancreatic Tumor (Label 2): Sphere inside/border of pancreas
    cz_t, cy_t, cx_t = 0.05, 0.15, 0.0
    tumor_mask = (
        ((grid_z - cz_t) / 0.12) ** 2 +
        ((grid_y - cy_t) / 0.12) ** 2 +
        ((grid_x - cx_t) / 0.12) ** 2
    ) <= 1.0
    
    # Tumor overwrites pancreas label
    label_volume[tumor_mask] = 2
    ct_volume[tumor_mask] = np.random.normal(loc=110.0, scale=20.0, size=tumor_mask.sum()).astype(np.float32)
    
    # Clip CT volume to reasonable HU range
    ct_volume = np.clip(ct_volume, -100.0, 240.0)
    
    return ct_volume, label_volume

def save_synthetic_dataset(output_dir="./dataset/Task07_Pancreas_Synthetic", num_samples=6, roi_size=(96, 96, 96)):
    import nibabel as nib
    
    out_path = Path(output_dir)
    img_dir = out_path / "imagesTr"
    lbl_dir = out_path / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_json = {
        "name": "Task07_Pancreas_Synthetic",
        "description": "Synthetic 3D Pancreas and Tumor CT benchmark dataset",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "pancreas", "2": "tumor"},
        "numTraining": num_samples,
        "training": []
    }
    
    print(f"[*] Generating {num_samples} synthetic CT volumes in '{output_dir}'...")
    affine = np.eye(4)
    
    for i in range(num_samples):
        sample_id = f"pancreas_{i+1:03d}"
        ct, lbl = generate_synthetic_ct_volume(spatial_shape=roi_size, seed=42 + i)
        
        img_nii = nib.Nifti1Image(ct, affine)
        lbl_nii = nib.Nifti1Image(lbl.astype(np.int16), affine)
        
        rel_img = f"imagesTr/{sample_id}.nii.gz"
        rel_lbl = f"labelsTr/{sample_id}.nii.gz"
        
        nib.save(img_nii, out_path / rel_img)
        nib.save(lbl_nii, out_path / rel_lbl)
        
        dataset_json["training"].append({
            "image": rel_img,
            "label": rel_lbl
        })
        print(f"  [+] Created {sample_id}.nii.gz")
        
    with open(out_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
        
    print(f"[*] Synthetic dataset created successfully with {num_samples} cases.")
    return str(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic 3D CT Dataset for PancreasGATUNet")
    parser.add_argument("--output_dir", type=str, default="./dataset/Task07_Pancreas_Synthetic", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of synthetic volumes")
    args = parser.parse_args()
    
    save_synthetic_dataset(output_dir=args.output_dir, num_samples=args.num_samples)
