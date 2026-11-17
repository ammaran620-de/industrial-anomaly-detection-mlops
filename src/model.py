"""PatchCore anomaly detection.

The idea in one paragraph: take a pretrained CNN, pull mid-level feature maps
for every normal training image, and store those patch features in a memory
bank. Mid-level layers matter -- early layers are too generic (edges, colour),
late layers are too tied to ImageNet classes. At inference time, a patch that
sits far from everything in the bank is anomalous. No anomalous training data
needed, which is the whole point for industrial inspection where defects are
rare and you can never enumerate them all.

Paper: Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (2022)

Checkpoints carry the whole configuration (layers, input size, preprocessing,
seed, feature size, optional calibrated threshold) and `load` refuses a
checkpoint that does not match the model or preprocessing it is asked to use.
Scores from a checkpoint served at the wrong input size are not comparable with
the ones it was evaluated at, so a mismatch is an error, not a warning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from src.data import preprocess_config

CHECKPOINT_FORMAT = 2
BACKBONE_NAME = "wide_resnet50_2"
WEIGHTS_NAME = "IMAGENET1K_V1"
MAX_POOL = 10000


class CheckpointError(ValueError):
    """The checkpoint cannot be used with the requested model or preprocessing."""


class PatchCore(nn.Module):
    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        input_size: int = 224,
        coreset_ratio: float = 0.01,
        n_neighbors: int = 9,
        device: str | None = None,
        seed: int | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = tuple(layers)
        self.input_size = input_size
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.weights_name = WEIGHTS_NAME if pretrained else None
        self.threshold: dict | None = None  # calibrated threshold, filled by train.py or load()

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = wide_resnet50_2(weights=weights)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.to(self.device)

        # Forward hooks capture intermediate feature maps without rewriting
        # the backbone's forward pass.
        self._features: dict[str, torch.Tensor] = {}
        for name in self.layers:
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
    def _coreset(features: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Greedy k-center subsampling.

        Keeping every patch would make the bank enormous and inference slow.
        Random sampling loses the rare-but-normal patches (an edge, a seam) that
        are exactly the ones that cause false positives when missing. k-center
        repeatedly picks the point furthest from everything chosen so far, so
        coverage is preserved at the configured fraction of the pool.
        """
        idx = [int(torch.randint(len(features), (1,), generator=generator).item())]
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
        # Both random steps (the pool subsample and the k-center start point)
        # draw from one generator, so the same seed and data give the same bank.
        # seed=None keeps the old unseeded behaviour.
        gen = None
        if self.seed is not None:
            gen = torch.Generator().manual_seed(int(self.seed))

        chunks = [self.extract(batch).cpu() for batch in loader]
        features = torch.cat(chunks, dim=0)

        # Greedy k-center is O(pool x keep) and the raw pool is ~200k patches,
        # which is hours on a CPU. A random pool of at most MAX_POOL patches is
        # drawn first, and coreset_ratio is applied to that pool, not to the full
        # set (0.05 of a 10,000 pool keeps 500 patches).
        if len(features) > MAX_POOL:
            idx = torch.randperm(len(features), generator=gen)[:MAX_POOL]
            features = features[idx]

        keep = max(1, int(len(features) * self.coreset_ratio))
        self.memory_bank = self._coreset(features.to(self.device), keep, gen).cpu()
        if len(self.memory_bank) < self.n_neighbors:
            raise ValueError(
                f"memory bank has {len(self.memory_bank)} patches but n_neighbors is "
                f"{self.n_neighbors}; raise coreset_ratio or add training images"
            )
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

    # ------------------------------------------------------------------ config

    def config(self) -> dict:
        """Everything needed to reproduce or validate this model, as plain Python types."""
        return {
            "backbone": BACKBONE_NAME,
            "weights": self.weights_name,
            "layers": list(self.layers),
            "input_size": int(self.input_size),
            "n_neighbors": int(self.n_neighbors),
            "coreset_ratio": float(self.coreset_ratio),
            "max_pool": MAX_POOL,
            "seed": self.seed,
            "feature_dim": int(self.memory_bank.shape[1]),
            "bank_size": int(self.memory_bank.shape[0]),
            "feature_map_shape": [int(v) for v in self.feature_map_shape],
            "preprocess": preprocess_config(self.input_size),
        }

    def save(self, path: str, threshold: dict | None = None) -> None:
        """Write the checkpoint. `threshold`, if given, is a calibrated operating point:
        {"value": float, "percentile": float, "n_calibration": int, "source": str}."""
        if self.memory_bank is None:
            raise RuntimeError("Nothing to save: call fit() first.")
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "memory_bank": self.memory_bank,
                "config": self.config(),
                "threshold": threshold,
            },
            path,
        )

    def load(self, path: str) -> "PatchCore":
        """Load a checkpoint into this model, rejecting any setting that does not match."""
        ckpt = _read_checkpoint(path)
        cfg = _normalised_config(ckpt)

        def need(name, have, want):
            if have != want:
                raise CheckpointError(
                    f"checkpoint {name} is {have!r} but this model was built with {want!r}; "
                    "scores would not be comparable"
                )

        need("input_size", cfg["input_size"], self.input_size)
        need("layers", tuple(cfg["layers"]), self.layers)
        need("backbone weights", cfg["weights"], self.weights_name)
        need("preprocessing", cfg["preprocess"], preprocess_config(self.input_size))

        bank = ckpt["memory_bank"]
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2:
            raise CheckpointError("memory_bank must be a 2-D tensor")
        if bank.shape[1] != cfg["feature_dim"]:
            raise CheckpointError(
                f"memory bank has feature size {bank.shape[1]} but the checkpoint records {cfg['feature_dim']}"
            )
        if len(bank) < cfg["n_neighbors"]:
            raise CheckpointError(
                f"memory bank has {len(bank)} patches, fewer than n_neighbors={cfg['n_neighbors']}"
            )

        # One dummy forward pass: the bank must match what this backbone really produces.
        with torch.no_grad():
            probe = self.extract(torch.zeros(1, 3, self.input_size, self.input_size))
        if probe.shape[1] != bank.shape[1]:
            raise CheckpointError(
                f"this backbone produces {probe.shape[1]}-dimensional patches but the bank holds "
                f"{bank.shape[1]}-dimensional ones"
            )
        if list(self.feature_map_shape) != list(cfg["feature_map_shape"]):
            raise CheckpointError(
                f"feature map is {list(self.feature_map_shape)} at this input size but the checkpoint "
                f"records {list(cfg['feature_map_shape'])}"
            )

        self.memory_bank = bank
        self.n_neighbors = int(cfg["n_neighbors"])
        self.coreset_ratio = float(cfg.get("coreset_ratio", self.coreset_ratio))
        self.seed = cfg.get("seed")
        self.threshold = ckpt.get("threshold")
        return self

    @classmethod
    def from_checkpoint(cls, path: str, input_size: int | None = None, device: str | None = None) -> "PatchCore":
        """Build a model from the checkpoint's own settings, then load it.

        If `input_size` is given and differs from the checkpoint, that is an error.
        """
        cfg = _normalised_config(_read_checkpoint(path))
        if input_size is not None and int(input_size) != cfg["input_size"]:
            raise CheckpointError(
                f"requested input size {input_size} but the checkpoint was built at {cfg['input_size']}"
            )
        model = cls(
            layers=tuple(cfg["layers"]), input_size=cfg["input_size"],
            coreset_ratio=cfg.get("coreset_ratio", 0.01), n_neighbors=cfg["n_neighbors"],
            device=device, seed=cfg.get("seed"), pretrained=cfg["weights"] is not None,
        )
        return model.load(path)


def _read_checkpoint(path: str) -> dict:
    # weights_only=True: a checkpoint is data, so it must not be able to run code when opened.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unpickling errors, truncated files, wrong file type
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(ckpt, dict) or "memory_bank" not in ckpt:
        raise CheckpointError(f"{path} is not a PatchCore checkpoint")
    return ckpt


def _normalised_config(ckpt: dict) -> dict:
    """Return the config for a current checkpoint, or reconstruct it for the original format.

    The original format stored layers, input_size, n_neighbors and feature_map_shape but no
    preprocessing, seed or threshold. It is accepted only where its recorded values can be checked.
    """
    fmt = ckpt.get("format")
    if fmt == CHECKPOINT_FORMAT:
        cfg = dict(ckpt.get("config") or {})
        missing = {"layers", "input_size", "n_neighbors", "weights", "preprocess", "feature_dim", "feature_map_shape"} - set(cfg)
        if missing:
            raise CheckpointError(f"checkpoint config is missing {sorted(missing)}")
        return cfg
    if fmt is None and {"layers", "input_size", "n_neighbors", "feature_map_shape"} <= set(ckpt):
        size = int(ckpt["input_size"])
        return {
            "layers": list(ckpt["layers"]), "input_size": size, "n_neighbors": int(ckpt["n_neighbors"]),
            "weights": WEIGHTS_NAME, "preprocess": preprocess_config(size),   # assumed: the original code only ever used these
            "feature_dim": int(ckpt["memory_bank"].shape[1]),
            "feature_map_shape": list(ckpt["feature_map_shape"]),
            "seed": None, "legacy": True,
        }
    raise CheckpointError(f"unsupported checkpoint format {fmt!r}")
