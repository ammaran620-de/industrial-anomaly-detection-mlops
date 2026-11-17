"""MVTec AD loading.

Structure on disk:
    data/<category>/train/good/*.png          <- normal only
    data/<category>/test/good/*.png
    data/<category>/test/<defect_type>/*.png
    data/<category>/ground_truth/<defect>/*_mask.png
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_config(size: int) -> dict:
    """The preprocessing the model was built with. Saved in every checkpoint and checked on load."""
    return {"resize": [int(size), int(size)], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


def build_transform(size: int = 224) -> transforms.Compose:
    cfg = preprocess_config(size)
    return transforms.Compose([
        transforms.Resize(tuple(cfg["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def split_fit_calibration(dataset: Dataset, calibration_fraction: float, seed: int):
    """Split normal training images into (fit, calibration) subsets.

    The memory bank is built from the fit subset only. The calibration subset is held out and used
    to choose the threshold, so neither the bank nor the final test images pick the operating point.
    The split depends only on `seed` and the dataset size.
    """
    n = len(dataset)
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")
    n_cal = int(round(n * calibration_fraction))
    if n_cal == 0:
        return torch.utils.data.Subset(dataset, list(range(n))), None
    if n_cal >= n:
        raise ValueError("calibration split would leave no images to fit on")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()
    cal, fit = sorted(order[:n_cal]), sorted(order[n_cal:])
    return torch.utils.data.Subset(dataset, fit), torch.utils.data.Subset(dataset, cal)


class MVTecDataset(Dataset):
    """split='train' yields tensors. split='test' yields (tensor, label, defect_type)."""

    def __init__(self, root: str, category: str, split: str = "train", size: int = 224):
        self.split = split
        self.transform = build_transform(size)
        base = Path(root) / category / split

        if not base.exists():
            raise FileNotFoundError(
                f"{base} not found. Run scripts/get_data.sh first."
            )

        self.samples: list[tuple[Path, int, str]] = []
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            label = 0 if folder.name == "good" else 1
            for img in sorted(folder.glob("*.png")):
                self.samples.append((img, label, folder.name))

        if not self.samples:
            raise RuntimeError(f"No images found under {base}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label, defect = self.samples[i]
        img = self.transform(Image.open(path).convert("RGB"))
        if self.split == "train":
            return img
        return img, label, defect


def make_loader_from(dataset: Dataset, batch_size: int = 8, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def make_loader(
    root: str, category: str, split: str = "train",
    batch_size: int = 8, size: int = 224, num_workers: int = 2,
) -> DataLoader:
    return make_loader_from(MVTecDataset(root, category, split, size), batch_size, num_workers)