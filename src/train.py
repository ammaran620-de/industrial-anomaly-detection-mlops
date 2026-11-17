"""Train PatchCore on one MVTec category and log everything to MLflow.

    python -m src.train --category carpet --coreset-ratio 0.05 --input-size 320 --seed 0

Normal training images are split into a fit set (builds the memory bank) and a held-out
calibration set (chooses the threshold), so the threshold is not taken from the test images
that the final evaluation scores. The same seed gives the same split and the same bank.

MLflow is the point of this file as much as the model is. Every run records its
params, its metrics and the resulting artifact, so three weeks later you can
answer "which coreset ratio gave the best AUROC?" without guessing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data import MVTecDataset, make_loader, make_loader_from, split_fit_calibration
from src.model import PatchCore


def evaluate(model: PatchCore, loader) -> dict[str, float]:
    scores, labels = [], []
    for imgs, lbls, _ in loader:
        s, _ = model.predict(imgs)
        scores.append(s)
        labels.append(lbls.numpy())

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "mean_score_normal": float(scores[labels == 0].mean()),
        "mean_score_anomalous": float(scores[labels == 1].mean()),
        "n_test": int(len(labels)),
    }


def calibrate_threshold(model: PatchCore, loader, percentile: float = 95.0) -> dict:
    """Threshold = a percentile of the anomaly scores of held-out normal images."""
    scores = np.concatenate([model.predict(batch)[0] for batch in loader])
    return {
        "value": float(np.percentile(scores, percentile)),
        "percentile": float(percentile),
        "n_calibration": int(len(scores)),
        "source": "held-out normal training images, excluded from the memory bank",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="carpet")
    p.add_argument("--data-root", default="data")
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--n-neighbors", type=int, default=9)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="seeds the random pool, the k-center start and the calibration split")
    p.add_argument("--calibration-fraction", type=float, default=0.1,
                   help="share of normal training images held out to choose the threshold (0 = no threshold)")
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument("--output", default="artifacts")
    p.add_argument("--experiment", default="patchcore-mvtec")
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.category}-cs{args.coreset_ratio}"):
        mlflow.log_params(vars(args))

        train_ds = MVTecDataset(args.data_root, args.category, "train", args.input_size)
        fit_ds, cal_ds = split_fit_calibration(train_ds, args.calibration_fraction, args.seed)
        train_loader = make_loader_from(fit_ds, args.batch_size)
        cal_loader = make_loader_from(cal_ds, args.batch_size) if cal_ds is not None else None
        test_loader = make_loader(
            args.data_root, args.category, "test",
            args.batch_size, args.input_size,
        )

        model = PatchCore(
            input_size=args.input_size,
            coreset_ratio=args.coreset_ratio,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )

        t0 = time.time()
        model.fit(train_loader)
        fit_seconds = time.time() - t0

        t0 = time.time()
        metrics = evaluate(model, test_loader)
        eval_seconds = time.time() - t0

        metrics["fit_seconds"] = fit_seconds
        metrics["eval_seconds"] = eval_seconds
        metrics["memory_bank_size"] = int(len(model.memory_bank))
        mlflow.log_metrics(metrics)

        threshold = None
        if cal_loader is not None:
            threshold = calibrate_threshold(model, cal_loader, args.threshold_percentile)
            threshold["calibration_fraction"] = args.calibration_fraction
            threshold["seed"] = args.seed
            mlflow.log_metrics({"threshold": threshold["value"], "n_calibration": threshold["n_calibration"]})
        else:
            print("[warn] --calibration-fraction 0: the checkpoint has no calibrated threshold")

        ckpt = Path(args.output) / f"patchcore_{args.category}.pt"
        model.save(str(ckpt), threshold=threshold)
        mlflow.log_artifact(str(ckpt))

        print(f"\n{args.category}")
        for k, v in metrics.items():
            print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
        if threshold:
            print(f"  threshold (p{threshold['percentile']:g} of {threshold['n_calibration']} held-out normals) {threshold['value']:.4f}")
        print(f"\nsaved -> {ckpt}")


if __name__ == "__main__":
    main()
