"""PatchCore anomaly detection.

The idea in one paragraph: take a pretrained CNN, pull mid-level feature maps
for every normal training image, and store those patch features in a memory
bank. Mid-level layers matter -- early layers are too generic (edges, colour),
late layers are too tied to ImageNet classes. At inference time, a patch that
sits far from everything in the bank is anomalous. No anomalous training data
needed, which is the whole point for industrial inspection where defects are
rare and you can never enumerate them all.

Paper: Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (2022)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class PatchCore(nn.Module):
    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        input_size: int = 224,
        coreset_ratio: float = 0.01,
        n_neighbors: int = 9,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.input_size = input_size
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors

        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.to(self.device)

        # Forward hooks capture intermediate feature maps without rewriting
        # the backbone's forward pass.
        self._features: dict[str, torch.Tensor] = {}
        for name in layers:
            getattr(self.backbone, name).register_forward_hook(self._hook(name))

        self.memory_bank: torch.Tensor | None = None
        self.feature_map_shape: tuple[int, int] | None = None

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            self._features[name] = out.detach()

        return fn

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings of shape (B * H * W, C)."""
        self._features.clear()
        self.backbone(x.to(self.device))

        maps = []
        target = None
        for name in self.layers:
            fmap = self._features[name]
            # Local average pooling smooths each patch against its neighbours,
            # which makes the bank far more robust to small misalignments.
            fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
            if target is None:
                target = fmap.shape[-2:]
            else:
                fmap = F.interpolate(fmap, size=target, mode="bilinear", align_corners=False)
            maps.append(fmap)

        merged = torch.cat(maps, dim=1)          # (B, C, H, W)
        self.feature_map_shape = tuple(merged.shape[-2:])
        b, c, h, w = merged.shape
        return merged.permute(0, 2, 3, 1).reshape(b * h * w, c)

    @staticmethod
    def _coreset(features: torch.Tensor, n: int) -> torch.Tensor:
        """Greedy k-center subsampling.

        Keeping every patch would make the bank enormous and inference slow.
        Random sampling loses the rare-but-normal patches (an edge, a seam) that
        are exactly the ones that cause false positives when missing. k-center
        repeatedly picks the point furthest from everything chosen so far, so
        coverage is preserved at ~1% of the size.
        """
        idx = [int(torch.randint(len(features), (1,)).item())]
        dist = torch.cdist(features, features[idx[-1]].unsqueeze(0)).squeeze(1)

        for _ in range(n - 1):
            nxt = int(torch.argmax(dist).item())
            idx.append(nxt)
            dist = torch.minimum(
                dist, torch.cdist(features, features[nxt].unsqueeze(0)).squeeze(1)
            )
        return features[idx]

    @torch.no_grad()
    def fit(self, loader) -> "PatchCore":
        chunks = [self.extract(batch).cpu() for batch in loader]
        features = torch.cat(chunks, dim=0)

        # Greedy k-center is O(pool x keep) and the raw pool is ~200k patches,
        # which is hours on a CPU. Narrowing the pool first keeps it tractable.
        max_pool = 10000
        if len(features) > max_pool:
            idx = torch.randperm(len(features))[:max_pool]
            features = features[idx]

        keep = max(1, int(len(features) * self.coreset_ratio))
        self.memory_bank = self._coreset(features.to(self.device), keep).cpu()
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (image_scores, anomaly_maps) for a batch."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() or load() before predict().")

        patches = self.extract(x)
        bank = self.memory_bank.to(self.device)

        dist = torch.cdist(patches, bank)
        knn, _ = dist.topk(self.n_neighbors, largest=False)
        scores = knn[:, 0]

        h, w = self.feature_map_shape
        b = x.shape[0]
        maps = scores.reshape(b, 1, h, w)
        maps = F.interpolate(
            maps, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )

        image_scores = maps.reshape(b, -1).max(dim=1).values
        return image_scores.cpu().numpy(), maps.squeeze(1).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "memory_bank": self.memory_bank,
                "feature_map_shape": self.feature_map_shape,
                "layers": self.layers,
                "input_size": self.input_size,
                "n_neighbors": self.n_neighbors,
            },
            path,
        )

    def load(self, path: str) -> "PatchCore":
        ckpt = torch.load(path, map_location="cpu")
        self.memory_bank = ckpt["memory_bank"]
        self.feature_map_shape = ckpt["feature_map_shape"]
        self.n_neighbors = ckpt["n_neighbors"]
        return self