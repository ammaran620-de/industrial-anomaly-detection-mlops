"""End-to-end through the real command-line entry points on a synthetic MVTec-layout dataset:

    src.train (seeded split, bank, calibration, save)  ->  src.evaluate (new run directory)  ->  src.api (lifespan, /predict)

The backbone is the tiny deterministic one from conftest.py, so this checks the wiring and the protocol, not
detection quality on real carpet images. No MVTec data or pretrained weights are used.
"""
import sys
from pathlib import Path

import httpx
import pytest
import torch

from src import api, evaluate, train
from tests.conftest import SIZE, anomalous_image, normal_image, png_bytes


def make_dataset(root: Path, category: str = "toy"):
    for i in range(20):
        _save(normal_image(i), root / category / "train" / "good" / f"{i:03d}.png")
    for i in range(200, 206):
        _save(normal_image(i), root / category / "test" / "good" / f"{i:03d}.png")
    for k in range(6):
        _save(anomalous_image(300 + k), root / category / "test" / "blob" / f"{k:03d}.png")


def _save(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def run_cli(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


@pytest.fixture
def trained(tiny_backbone, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    make_dataset(tmp_path / "data")
    out = tmp_path / "artifacts"
    run_cli(monkeypatch, train, "--category", "toy", "--data-root", str(tmp_path / "data"), "--output", str(out),
            "--input-size", str(SIZE), "--coreset-ratio", "0.2", "--n-neighbors", "3", "--seed", "5",
            "--calibration-fraction", "0.25", "--batch-size", "4")
    return {"root": tmp_path, "ckpt": out / "patchcore_toy.pt"}


def test_train_writes_a_checkpoint_with_a_held_out_calibration(trained):
    ckpt = torch.load(trained["ckpt"], weights_only=True)
    assert ckpt["format"] == 2 and ckpt["config"]["seed"] == 5 and ckpt["config"]["input_size"] == SIZE
    thr = ckpt["threshold"]
    assert thr["n_calibration"] == 5 and thr["percentile"] == 95.0     # 25% of 20 training images
    assert "excluded from the memory bank" in thr["source"] and thr["seed"] == 5


def test_evaluate_uses_the_checkpoint_threshold_and_never_overwrites(trained, monkeypatch):
    reports = trained["root"] / "reports"
    old = reports / "toy"
    old.mkdir(parents=True)
    (old / "metrics.txt").write_text("recorded earlier")
    args = ("--category", "toy", "--data-root", str(trained["root"] / "data"), "--checkpoint", str(trained["ckpt"]),
            "--out", str(reports), "--n-vis", "2")
    run_cli(monkeypatch, evaluate, *args, "--run-name", "first")
    run_cli(monkeypatch, evaluate, *args, "--run-name", "second")
    text = (old / "runs" / "first" / "metrics.txt").read_text()
    assert "independent of the test set" in text and "image AUROC       1.0000" in text
    assert (old / "runs" / "second" / "per_image.csv").exists()
    assert (old / "metrics.txt").read_text() == "recorded earlier"
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, evaluate, *args, "--run-name", "first")


def test_evaluate_rejects_an_input_size_that_differs_from_the_checkpoint(trained, monkeypatch):
    from src.model import CheckpointError
    with pytest.raises(CheckpointError, match="requested input size 96"):
        run_cli(monkeypatch, evaluate, "--category", "toy", "--data-root", str(trained["root"] / "data"),
                "--checkpoint", str(trained["ckpt"]), "--out", str(trained["root"] / "r"), "--input-size", "96")


def test_same_seed_reproduces_the_checkpoint(trained, tmp_path, monkeypatch):
    again = tmp_path / "again"
    run_cli(monkeypatch, train, "--category", "toy", "--data-root", str(tmp_path / "data"), "--output", str(again),
            "--input-size", str(SIZE), "--coreset-ratio", "0.2", "--n-neighbors", "3", "--seed", "5",
            "--calibration-fraction", "0.25", "--batch-size", "4")
    a, b = (torch.load(p, weights_only=True) for p in (trained["ckpt"], again / "patchcore_toy.pt"))
    assert torch.equal(a["memory_bank"], b["memory_bank"])
    assert a["threshold"]["value"] == b["threshold"]["value"]


@pytest.mark.asyncio
async def test_trained_checkpoint_is_served_and_separates_the_classes(trained, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", str(trained["ckpt"]))
    monkeypatch.setattr(api, "INPUT_SIZE", None)
    monkeypatch.setattr(api, "THRESHOLD", None)
    try:
        async with api.app.router.lifespan_context(api.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t") as c:
                good = await c.post("/predict", files={"file": ("g.png", png_bytes(normal_image(900)), "image/png")})
                bad = await c.post("/predict", files={"file": ("b.png", png_bytes(anomalous_image(901)), "image/png")})
        assert good.status_code == bad.status_code == 200
        assert good.json()["is_anomalous"] is False and bad.json()["is_anomalous"] is True
    finally:
        api.model = api.transform = api.threshold = api.threshold_source = None
        api.recent_scores.clear()
