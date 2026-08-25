"""Train PatchCore on one MVTec category and log everything to MLflow.

    python -m src.train --category carpet --coreset-ratio 0.01

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

from src.data import make_loader
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="carpet")
    p.add_argument("--data-root", default="data")
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--n-neighbors", type=int, default=9)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="artifacts")
    p.add_argument("--experiment", default="patchcore-mvtec")
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.category}-cs{args.coreset_ratio}"):
        mlflow.log_params(vars(args))

        train_loader = make_loader(
            args.data_root, args.category, "train",
            args.batch_size, args.input_size,
        )
        test_loader = make_loader(
            args.data_root, args.category, "test",
            args.batch_size, args.input_size,
        )

        model = PatchCore(
            input_size=args.input_size,
            coreset_ratio=args.coreset_ratio,
            n_neighbors=args.n_neighbors,
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

        ckpt = Path(args.output) / f"patchcore_{args.category}.pt"
        model.save(str(ckpt))
        mlflow.log_artifact(str(ckpt))

        print(f"\n{args.category}")
        for k, v in metrics.items():
            print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
        print(f"\nsaved -> {ckpt}")


if __name__ == "__main__":
    main()
