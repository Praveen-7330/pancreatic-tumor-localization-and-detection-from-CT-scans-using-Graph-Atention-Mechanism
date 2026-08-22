import os
import json
from pathlib import Path
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, list_data_collate
from data.transforms import get_train_transforms, get_val_transforms

def load_msd_pancreas_files(dataset_dir):
    json_path = Path(dataset_dir) / "dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    train_entries = data.get("training", [])
    data_dicts = []
    for item in train_entries:
        img_path = str(Path(dataset_dir) / item["image"])
        lbl_path = str(Path(dataset_dir) / item["label"])
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_dicts.append({"image": img_path, "label": lbl_path})
    return data_dicts

def get_dataloaders(data_dicts, fold=0, n_splits=5, roi_size=(96, 96, 96), batch_size=2, num_workers=2, cache_rate=0.5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(data_dicts))
    train_idx, val_idx = splits[fold]
    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]
    train_transforms = get_train_transforms(roi_size=roi_size)
    val_transforms = get_val_transforms()
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=list_data_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
