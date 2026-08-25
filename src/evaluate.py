"""Full evaluation: metrics, per-image CSV, and bounding-box visualisations.

    python -m src.evaluate --category carpet

Produces, under reports/<category>/:
    metrics.txt      the numbers, including a per-defect-type breakdown
    per_image.csv    one row per test image: path, label, score, verdict
    vis/*.png        original | heatmap | boxes, for a sample of images

Why this exists: AUROC alone tells you the ranking is good, not whether the
system works at a chosen threshold. Precision and recall at a real threshold
are what an operator on a line actually experiences.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data import MVTecDataset, build_transform
from src.model import PatchCore


def collect(model: PatchCore, ds: MVTecDataset, batch_size: int = 8):
    """Run the whole test set, keeping scores, maps and metadata aligned."""
    scores, labels, defects, paths, maps = [], [], [], [], []

    for start in range(0, len(ds), batch_size):
        batch_items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        imgs = torch.stack([b[0] for b in batch_items])

        s, m = model.predict(imgs)
        scores.extend(s.tolist())
        maps.extend(list(m))

        for j, b in enumerate(batch_items):
            labels.append(b[1])
            defects.append(b[2])
            paths.append(str(ds.samples[start + j][0]))

        print(f"  scored {min(start + batch_size, len(ds))}/{len(ds)}", end="\r")

    print()
    return np.array(scores), np.array(labels), defects, paths, maps


def pick_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (percentile_threshold, best_f1_threshold).

    The percentile rule is what you'd use in practice: you only have normal
    data at deployment time, so you set the bar just above what normal looks
    like. The best-F1 value uses the test labels, so it is an upper bound you
    could not actually achieve in production -- useful as a reference only.
    """
    normal = scores[labels == 0]
    percentile_thr = float(np.percentile(normal, 95))

    best_f1, best_thr = -1.0, percentile_thr
    for thr in np.linspace(scores.min(), scores.max(), 300):
        pred = (scores > thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return percentile_thr, best_thr


def boxes_from_map(amap: np.ndarray, thr: float, min_area: int = 80):
    """Threshold the anomaly map, then bound each connected blob.

    Connected-component labelling is enough here -- defects are contiguous
    regions, so anything fancier would be over-engineering.
    """
    mask = amap > thr
    labelled, n = ndimage.label(mask)
    out = []
    for sl in ndimage.find_objects(labelled):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        if (y1 - y0) * (x1 - x0) >= min_area:
            out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def visualise(path: str, amap: np.ndarray, score: float, defect: str,
              flagged: bool, pixel_thr: float, out_path: Path, size: int) -> None:
    from PIL import Image

    img = np.array(Image.open(path).convert("RGB").resize((size, size)))
    boxes = boxes_from_map(amap, pixel_thr)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    axes[0].imshow(img)
    axes[0].set_title(f"input — {defect}", fontsize=11)

    axes[1].imshow(img)
    axes[1].imshow(amap, cmap="jet", alpha=0.5)
    axes[1].set_title("anomaly heatmap", fontsize=11)

    axes[2].imshow(img)
    for (x, y, w, h) in boxes:
        axes[2].add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2.2)
        )
    verdict = "FLAGGED" if flagged else "passed"
    axes[2].set_title(
        f"{verdict} — score {score:.2f} — {len(boxes)} region(s)", fontsize=11
    )

    for a in axes:
        a.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="carpet")
    p.add_argument("--data-root", default="data")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--n-vis", type=int, default=12)
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    ckpt = args.checkpoint or f"artifacts/patchcore_{args.category}.pt"
    out_dir = Path(args.out) / args.category
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {ckpt}")
    model = PatchCore(input_size=args.input_size).load(ckpt)

    ds = MVTecDataset(args.data_root, args.category, "test", args.input_size)
    print(f"scoring {len(ds)} test images")
    scores, labels, defects, paths, maps = collect(model, ds)

    auroc = roc_auc_score(labels, scores)
    pct_thr, f1_thr = pick_threshold(scores, labels)
    thr = args.threshold if args.threshold is not None else pct_thr

    preds = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    accuracy = (tp + tn) / len(labels)

    lines = []
    add = lines.append
    add(f"category            {args.category}")
    add(f"checkpoint          {ckpt}")
    add(f"test images         {len(labels)}  ({int((labels==0).sum())} good, "
        f"{int((labels==1).sum())} defective)")
    add("")
    add("threshold-free")
    add(f"  image AUROC       {auroc:.4f}")
    add("")
    add(f"at threshold {thr:.4f}   (95th pct of normal scores)")
    add(f"  accuracy          {accuracy:.4f}")
    add(f"  precision         {precision:.4f}   of flagged items, this fraction were real defects")
    add(f"  recall            {recall:.4f}   of real defects, this fraction were caught")
    add(f"  F1                {f1:.4f}")
    add("")
    add("  confusion matrix")
    add(f"    true negative   {tn:4d}   good, passed")
    add(f"    false positive  {fp:4d}   good, wrongly flagged")
    add(f"    false negative  {fn:4d}   defective, missed")
    add(f"    true positive   {tp:4d}   defective, caught")
    add("")
    add(f"best achievable F1 threshold {f1_thr:.4f} (uses test labels - reference only)")
    add("")
    add("score distribution")
    add(f"  normal      mean {scores[labels==0].mean():.4f}   "
        f"min {scores[labels==0].min():.4f}   max {scores[labels==0].max():.4f}")
    add(f"  defective   mean {scores[labels==1].mean():.4f}   "
        f"min {scores[labels==1].min():.4f}   max {scores[labels==1].max():.4f}")
    add("")
    add("per defect type")
    for d in sorted(set(defects)):
        idx = [i for i, x in enumerate(defects) if x == d]
        sub = scores[idx]
        caught = int((sub > thr).sum())
        tag = "passed" if d == "good" else "caught"
        add(f"  {d:14s} n={len(idx):3d}   mean {sub.mean():.4f}   "
            f"{tag} {caught if d != 'good' else len(idx)-caught}/{len(idx)}")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "metrics.txt").write_text(report, encoding="utf-8")

    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "defect_type", "true_label", "score", "predicted_anomalous", "correct"])
        for pth, d, lb, sc, pr in zip(paths, defects, labels, scores, preds):
            w.writerow([pth, d, int(lb), f"{sc:.6f}", int(pr), int(lb == pr)])

    # Pixel threshold for the boxes: high percentile across all maps, so only
    # the strongest regions get boxed rather than every warm patch.
    all_vals = np.concatenate([m.ravel() for m in maps[: min(40, len(maps))]])
    pixel_thr = float(np.percentile(all_vals, 99))

    picks = [i for i, l in enumerate(labels) if l == 1][: args.n_vis - 3]
    picks += [i for i, l in enumerate(labels) if l == 0][:3]

    print(f"\nwriting {len(picks)} visualisations")
    for i in picks:
        name = f"{defects[i]}_{Path(paths[i]).stem}.png"
        visualise(paths[i], maps[i], scores[i], defects[i],
                  bool(preds[i]), pixel_thr, vis_dir / name, args.input_size)

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
