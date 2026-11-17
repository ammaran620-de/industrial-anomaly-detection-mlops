# Industrial Anomaly Detection — PatchCore with a Production Wrapper

Unsupervised surface defect detection, trained on MVTec AD and served behind a
containerised API with experiment tracking and a basic drift signal.

**99.56% image AUROC on carpet. 87 of 89 defects caught at the operating threshold.**

The model half is PatchCore. The other half is everything needed before a model
is useful to anyone: reproducible runs, a serving layer, an evaluation report,
tests, and a way to notice when the input distribution moves.

---

## Why unsupervised

Supervised detection needs labelled examples of every defect you care about. On
a production line that assumption breaks immediately — defects are rare, the
dangerous ones are the ones nobody has seen before, and by the time you have a
hundred labelled examples of a defect you have already shipped a hundred bad
units.

PatchCore trains only on **normal** images. It builds a memory bank of what
normal looks like at patch level, then flags anything far from it. Novel defect
types are caught by construction rather than by anticipation.

---

## How it works

1. A frozen ImageNet-pretrained WideResNet-50 extracts feature maps from
   `layer2` and `layer3`. Mid-level on purpose — early layers are generic edges
   and colour, late layers are tied to ImageNet's class semantics.
2. Each patch is average-pooled against its 3x3 neighbourhood, making the
   representation tolerant of small misalignments.
3. Patches from every normal training image go into a memory bank.
4. **Coreset subsampling** cuts the bank to ~1-5% via greedy k-center.
5. At inference, each patch's distance to its nearest neighbour in the bank is
   its score. Image score is the max over patches; the upsampled grid is a
   heatmap showing *where*.

Nothing is trained. The backbone is frozen and there is no loss function —
"training" here means *remembering* what normal looks like.

### Why coreset instead of random sampling

The full bank is large and slow to search. Random sampling shrinks it but
discards the rare-but-normal patches — a seam, a fabric edge, a printed
marking. Those are exactly the patches that cause false positives when absent,
because a legitimate seam then resembles nothing in the bank. Greedy k-center
picks points maximising coverage, preserving that tail.

---

## Results — MVTec AD, carpet

### Configuration comparison

| Coreset ratio | Input size | Bank size | AUROC | Fit time |
|---|---|---|---|---|
| 0.01 | 224 | 100 | 0.9795 | 106 s |
| 0.05 | 224 | 500 | 0.9916 | 172 s |
| **0.05** | **320** | **500** | **0.9956** | **277 s** |

A 5x larger memory bank buys 1.2 points of AUROC. Higher resolution buys
another 0.4, at 1.6x the fit time. All runs on CPU, no GPU.

### Performance at the operating threshold

Threshold set to the **95th percentile of normal scores (2.1189)** — the rule
you can actually apply in production, where only normal data is available.

| Metric | Value |
|---|---|
| Accuracy | 0.9658 |
| Precision | 0.9775 |
| Recall | 0.9775 |
| F1 | 0.9775 |

|  | Passed | Flagged |
|---|---|---|
| **Good (28)** | 26 | 2 |
| **Defective (89)** | 2 | 87 |

### Per defect type

| Defect | n | Mean score | Caught |
|---|---|---|---|
| hole | 17 | 3.03 | 17/17 |
| cut | 17 | 2.97 | 17/17 |
| metal contamination | 17 | 2.75 | 17/17 |
| thread | 19 | 2.75 | 17/19 |
| color | 19 | 2.71 | 19/19 |
| good | 28 | 1.86 | 26/28 passed |

**Both misses are thread defects.** That is consistent with the physics: a
stray thread is thin and low-contrast against woven carpet, so it barely
disturbs local patch statistics. Holes and cuts destroy the texture outright
and score highest.

### Score separation

| | Mean | Min | Max |
|---|---|---|---|
| Normal | 1.8626 | 1.6671 | 2.1772 |
| Defective | 2.8365 | 2.0255 | 3.5093 |

The distributions overlap between 2.03 and 2.18 — which is why perfect
separation is impossible and why the threshold is a genuine decision rather
than a formality.

---

## Threshold selection is a business decision

The costs are asymmetric. A missed defect reaches a customer. A false alarm
costs an operator thirty seconds of inspection. So in a real deployment the
threshold should sit **below** the balanced point, accepting more false
positives to reduce misses.

The 95th-percentile rule above is deliberately the version that works with
normal data only. The best-achievable F1 threshold (2.0183) requires test
labels and could not be computed in production — it is reported as a reference
bound, not a recommendation.

---

## Running it

```bash
pip install -r requirements.txt

# MVTec AD requires registration: mvtec.com/company/research/datasets/mvtec-ad
# Extract a category into ./data, then:
python -m src.train --category carpet --coreset-ratio 0.05 --input-size 320
python -m src.evaluate --category carpet --input-size 320

python -m mlflow ui                          # runs at localhost:5000
python -m uvicorn src.api:app --reload       # API at localhost:8000/docs
```

The trained checkpoint (`artifacts/patchcore_carpet.pt`, or the path in `MODEL_PATH`) is not committed. Until `src.train` has produced it, the server starts but `/health` reports `model_loaded: false` and `/predict` returns 503. A healthy server on its own does not mean inference works.

With Docker:

```bash
docker compose up --build
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, and whether a model is loaded |
| `POST /predict` | Multipart image to score, verdict, latency |
| `GET /stats` | Rolling mean, p95 and anomaly rate over recent requests |

`/stats` is a cheap drift signal. If the live mean drifts from the validation
value, the input distribution has moved — new material, changed lighting, a
nudged camera — and the threshold needs revisiting.

---

## Known limitations

**Latency.** 535 ms per image on CPU. A line running at 30 fps needs 33 ms.
Closing that gap needs ONNX export, quantisation, and GPU or edge accelerator
inference — not addressed here.

**Thread defects.** 2/19 missed. Low-contrast linear defects are the weak spot;
higher resolution or a defect-specific supervised head would help.

**Single category.** Tuned on carpet only. Other MVTec categories need their
own threshold, and probably their own coreset ratio.

**Drift detection is rudimentary.** A rolling mean catches gross shifts, not
gradual ones. Proper distribution-distance testing is the right fix.

---

## What I'd do next

- Pixel-level AUROC and PRO score, not just image-level
- ONNX export plus quantisation for edge deployment
- Distribution-distance drift detection replacing the rolling window
- Compare against PaDiM and EfficientAD on the same split
- Per-category threshold calibration across all fifteen categories

---

## Layout

```
src/model.py     PatchCore: feature extraction, coreset, scoring
src/data.py      MVTec loading
src/train.py     Training + evaluation, logged to MLflow
src/evaluate.py  Full metrics report, per-image CSV, box visualisations
src/api.py       FastAPI serving
tests/           API tests, run in CI
reports/         Generated metrics and visualisations
Dockerfile       CPU-only image
```

---

## Reference

Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
Dataset: MVTec AD (Bergmann et al.), CC BY-NC-SA 4.0, non-commercial research use.
