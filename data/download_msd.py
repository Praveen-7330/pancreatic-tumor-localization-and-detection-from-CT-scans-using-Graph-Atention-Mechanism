import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

MSD_PANCREAS_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_msd_pancreas(target_dir="./dataset"):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    tar_path = target_path / "Task07_Pancreas.tar"
    extracted_dir = target_path / "Task07_Pancreas"
    
    if extracted_dir.exists():
        print(f"[INFO] Dataset already exists at: {extracted_dir.resolve()}")
        return str(extracted_dir)
        
    if not tar_path.exists():
        print(f"[INFO] Downloading MSD Task07 Pancreas from {MSD_PANCREAS_URL}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="MSD Pancreas") as t:
            urllib.request.urlretrieve(MSD_PANCREAS_URL, filename=str(tar_path), reporthook=t.update_to)
        print("[INFO] Download completed.")
        
    print(f"[INFO] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)
    print(f"[INFO] Extraction complete at: {extracted_dir.resolve()}")
    return str(extracted_dir)

if __name__ == "__main__":
    download_and_extract_msd_pancreas()
