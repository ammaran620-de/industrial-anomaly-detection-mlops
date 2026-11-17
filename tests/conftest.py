"""Shared fixtures: a tiny deterministic backbone so the whole PatchCore path (fit, calibrate, save, load,
serve) runs in a second with no downloads. The real wide_resnet50_2 architecture is exercised separately
in test_model_config.py with random weights."""
import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

import src.model as model_module
from src.data import build_transform
from src.model import PatchCore

SIZE = 64


class TinyBackbone(nn.Module):
    """Has the layer2/layer3 modules PatchCore hooks. Seeded, so features are the same every run."""

    def __init__(self):
        super().__init__()
        g = torch.manual_seed(1234)  # noqa: F841  (global seed set for the layer init below)
        self.stem = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.layer2 = nn.Sequential(nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, x):
        return self.layer3(self.layer2(F.relu(self.stem(x))))


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr(model_module, "wide_resnet50_2", lambda weights=None: TinyBackbone())


def normal_image(i: int, size: int = SIZE) -> Image.Image:
    """A smooth mid-grey texture with a little seeded noise: stands in for a defect-free sample."""
    rng = np.random.default_rng(i)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 128 + 14 * np.sin((xx + i) / 5.0) + 10 * np.cos((yy + 2 * i) / 7.0)
    img = np.clip(base + rng.normal(0, 3, (size, size)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([img] * 3, axis=-1))


def anomalous_image(i: int = 99, size: int = SIZE) -> Image.Image:
    """The same texture with a bright square: a defect no normal training image contains."""
    arr = np.array(normal_image(i, size))
    arr[20:44, 20:44] = 255
    return Image.fromarray(arr)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_batch(images, size: int = SIZE) -> torch.Tensor:
    tf = build_transform(size)
    return torch.stack([tf(im) for im in images])


@pytest.fixture
def fitted(tiny_backbone):
    """(model, fit_images, calibration_images) with a bank built from the fit images only."""
    fit_imgs = [normal_image(i) for i in range(12)]
    cal_imgs = [normal_image(i) for i in range(100, 108)]
    model = PatchCore(input_size=SIZE, coreset_ratio=0.2, n_neighbors=3, device="cpu", seed=0, pretrained=False)
    model.fit([to_batch(fit_imgs[:6]), to_batch(fit_imgs[6:])])
    return model, fit_imgs, cal_imgs


@pytest.fixture
def checkpoint(fitted, tmp_path):
    """A saved checkpoint with a calibrated threshold (p95 of the held-out normal images)."""
    model, _, cal_imgs = fitted
    scores, _ = model.predict(to_batch(cal_imgs))
    threshold = {"value": float(np.percentile(scores, 95)), "percentile": 95.0,
                 "n_calibration": len(cal_imgs), "source": "held-out normal images"}
    path = tmp_path / "patchcore_test.pt"
    model.save(str(path), threshold=threshold)
    return {"path": str(path), "threshold": threshold, "cal_scores": scores, "model": model}
