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


def build_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


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


def make_loader(
    root: str, category: str, split: str = "train",
    batch_size: int = 8, size: int = 224, num_workers: int = 2,
) -> DataLoader:
    ds = MVTecDataset(root, category, split, size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )