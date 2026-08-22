import numpy as np
from scipy.ndimage import label

def filter_small_connected_components(mask, min_size=50):
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if component.sum() >= min_size:
            cleaned_mask[component] = 1
    return cleaned_mask

def extract_3d_bounding_box(binary_mask):
    if binary_mask.sum() == 0:
        return None, None
    pos = np.where(binary_mask > 0)
    z_min, z_max = int(np.min(pos[0])), int(np.max(pos[0]))
    y_min, y_max = int(np.min(pos[1])), int(np.max(pos[1]))
    x_min, x_max = int(np.min(pos[2])), int(np.max(pos[2]))
    centroid = (float(np.mean(pos[0])), float(np.mean(pos[1])), float(np.mean(pos[2])))
    bbox = {
        "z_min": z_min, "z_max": z_max,
        "y_min": y_min, "y_max": y_max,
        "x_min": x_min, "x_max": x_max,
        "volume_voxels": int(binary_mask.sum()),
        "centroid": centroid
    }
    return bbox, centroid
