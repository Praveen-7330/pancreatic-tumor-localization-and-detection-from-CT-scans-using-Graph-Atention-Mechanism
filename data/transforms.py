from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandAffined, RandFlipd, RandGaussianNoised, RandGaussianSmoothd,
    EnsureTyped, CastToTyped
)
import torch

def get_train_transforms(roi_size=(96, 96, 96), hu_min=-100.0, hu_max=240.0, num_samples=4):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=roi_size,
            pos=2.0, neg=1.0, num_samples=num_samples, image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        CastToTyped(keys=["label"], dtype=torch.int64),
        EnsureTyped(keys=["image", "label"])
    ])

def get_val_transforms(hu_min=-100.0, hu_max=240.0):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CastToTyped(keys=["label"], dtype=torch.int64),
        EnsureTyped(keys=["image", "label"])
    ])
