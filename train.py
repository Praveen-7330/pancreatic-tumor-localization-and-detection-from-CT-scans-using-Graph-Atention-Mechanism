import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

from data.dataset import load_msd_pancreas_files, get_dataloaders
from data.synthetic_generator import save_synthetic_dataset
from data.download_msd import download_and_extract_msd_pancreas
from models.full_gat_unet import PancreasGATUNet
from training.trainer import PancreasTrainer
from utils.device_utils import get_device
from utils.path_utils import resolve_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train PancreasGATUNet Model on MSD Task07 Pancreas CT Dataset")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic 3D CT dataset for dry-run/testing")
    parser.add_argument("--download", action="store_true", help="Download official MSD Task07 dataset if not present")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Override config params from CLI
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
        
    device = get_device()
    print(f"==================================================")
    print(f"  PancreasGATUNet Master's Mini Project Trainer")
    print(f"==================================================")
    print(f"[*] Compute Device: {device}")
    
    # Dataset Handling
    if args.synthetic:
        data_dir = resolve_path('dataset', 'Task07_Pancreas_Synthetic')
        if not data_dir.exists() or len(list(data_dir.iterdir())) == 0:
            print("[*] Generating synthetic benchmark dataset...")
            save_synthetic_dataset(output_dir=data_dir, num_samples=6)
        config["data"]["data_dir"] = data_dir
    else:
        data_dir = config["data"]["data_dir"]
        if not os.path.exists(data_dir) and args.download:
            print("[*] Downloading official MSD Task07 dataset...")
            download_and_extract_msd_pancreas(target_dir="./dataset")
        elif not os.path.exists(data_dir):
            print(f"[WARNING] Dataset not found at '{data_dir}'. Switching to synthetic dry-run dataset.")
            data_dir = "./dataset/Task07_Pancreas_Synthetic"
            if not os.path.exists(data_dir):
                save_synthetic_dataset(output_dir=data_dir, num_samples=6)
            config["data"]["data_dir"] = data_dir

    print(f"[*] Loading data entries from: {config['data']['data_dir']}")
    data_dicts = load_msd_pancreas_files(config["data"]["data_dir"])
    print(f"[*] Found {len(data_dicts)} 3D CT volumes.")
    
    train_loader, val_loader = get_dataloaders(
        data_dicts,
        fold=0,
        n_splits=5,
        roi_size=tuple(config["preprocessing"]["roi_size"]),
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        cache_rate=0.5
    )
    
    # Initialize Model
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
    )
    
    print("[*] PancreasGATUNet model initialized successfully.")
    
    # Trainer Setup
    trainer = PancreasTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Execute Training
    trainer.fit(num_epochs=config["training"]["epochs"])

if __name__ == "__main__":
    main()
