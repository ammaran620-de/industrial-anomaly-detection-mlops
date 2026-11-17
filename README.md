# Industrial Anomaly Detection — PatchCore with a Production Wrapper

Unsupervised surface defect detection, trained on MVTec AD and served behind a
containerised API with experiment tracking and a basic drift signal.

**Recorded run on carpet: 99.56% image AUROC; 87 of 89 defects caught at a threshold taken from the test set's own good images.**
These figures come from the original run, before checkpoint validation, seeding and a held-out calibration split
were added. They are kept as recorded and labelled below; see [Evaluation protocol](#evaluation-protocol).

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
4. **Coreset subsampling** shrinks the bank in two steps (see `PatchCore.fit`):
   the patches are first randomly subsampled to a pool of at most 10,000 (the
   full set is about 200,000), then greedy k-center keeps `coreset_ratio` of that
   pool. A ratio of 0.05 therefore keeps 500 patches.
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

The headline 0.9956 comes from the last row only: coreset ratio 0.05, 320 px
input, 500-patch bank. The first row (0.01, 224 px) is what `src.train` uses if
you pass no flags. These runs predate seeding: the random pool step was not
seeded, so each figure is one recorded run per configuration and a rerun can
differ. The recorded output for the headline run is in
`reports/carpet/metrics.txt`, and `reports/carpet/HISTORICAL.md` says exactly how
it was produced.

### Evaluation protocol

| | Recorded run (the tables here) | Current code (`src.train`, `src.evaluate`) |
|---|---|---|
| Threshold | 95th percentile of the 28 good **test** images | 95th percentile of held-out **normal training** images, excluded from the memory bank |
| Independent of the test set | No: about 5% of those good images are flagged by construction | Yes |
| Random seed | None | `--seed`, default 0; seeds the pool subsample, the k-center start and the calibration split |
| Checkpoint | Weights only; nothing checked on load | Records config, preprocessing, seed and threshold; refuses to load if any of them disagrees with the model or request |
| Output directory | `reports/carpet/` (overwritten each run) | `reports/<category>/runs/<run-name>/`, a new directory every run |

No run with the current protocol is reported here. It needs the MVTec AD carpet
data and the pretrained WideResNet-50 weights, and neither was reachable when this
was written, so every number in this README is from the recorded run. Rerun the
commands below to get seeded, calibrated figures; they will not necessarily equal
these.

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
normal data only. In the recorded run it was computed from the 28 good test
images, so the same images set the threshold and scored it, and about 5% of them
(here 2 of 28) were flagged by construction. The current `src.train` instead holds
out a fraction of the normal training images (`--calibration-fraction`, default
0.1), builds the memory bank from the rest, and stores the 95th percentile of the
held-out scores in the checkpoint. `src.evaluate` and the API use that stored
value, so the test set plays no part in choosing it. A checkpoint without a
calibration falls back to the old test-derived rule, and the report says so. The
best-achievable F1 threshold (2.0183 in the recorded run) requires test labels and
could not be computed in production: it is a reference bound, not a recommendation.

---

## Running it

```bash
pip install -r requirements.txt

# MVTec AD requires registration: mvtec.com/company/research/datasets/mvtec-ad
# Extract a category into ./data, then:
python -m src.train --category carpet --coreset-ratio 0.05 --input-size 320 --seed 0
python -m src.evaluate --category carpet --run-name seed0

python -m mlflow ui                          # runs at localhost:5000
python -m uvicorn src.api:app --reload       # API at localhost:8000/docs
```

The trained checkpoint (`artifacts/patchcore_carpet.pt`, or the path in `MODEL_PATH`) is not committed. Until `src.train` has produced it, the server starts but `/health` reports `model_loaded: false` and `/predict` returns 503. A healthy server on its own does not mean inference works.

`src.train` defaults to 0.01 and 224 px, so pass the flags above to reproduce the
0.05 / 320 px configuration. Nothing else needs to match: the checkpoint records its
own input size, layers, preprocessing and threshold, and `src.evaluate` and the API
read them from it. If you pass `--input-size` to `src.evaluate` or set `INPUT_SIZE`
for the API, the value must equal the checkpoint's; a different one is an error, not a
silent resize. `THRESHOLD` is an optional override; unset, the calibrated value in
the checkpoint is used.

The memory bank is `coreset_ratio` times a pool of at most 10,000 randomly chosen
patches, so a ratio of 0.05 gives 500 patches and 0.01 gives 100.

With Docker:

```bash
docker compose up --build
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, whether a model is loaded, and if so its input size, threshold, where the threshold came from, and seed |
| `POST /predict` | Multipart image to score, verdict, latency |
| `GET /stats` | Rolling mean, p95 and anomaly rate over recent requests |

The checkpoint is loaded once, at startup. A missing file leaves the server up with
no model (`/predict` returns 503). A checkpoint that disagrees with the code or with
`INPUT_SIZE`, that cannot be read safely, or that has no threshold when `THRESHOLD`
is unset stops the server with an error naming the mismatch, rather than serving
scores that mean something else. Checkpoints are opened with `weights_only=True`.

`/stats` is a cheap drift signal. If the live mean drifts from the validation
value, the input distribution has moved — new material, changed lighting, a
nudged camera — and the threshold needs revisiting.

---

## Known limitations

**Latency.** Not measured in this repository. An earlier version of this README
gave 535 ms per image on CPU, but no script or recorded output here supports
that number, so do not treat it as a result. `/predict` returns a `latency_ms`
field timed around preprocessing and scoring (not model loading), so you can
measure it on your own machine once a checkpoint exists. A line running at
30 fps needs 33 ms; closing that gap would need ONNX export, quantisation, and
GPU or edge accelerator inference, none of which is addressed here.

**Thread defects.** 2/19 missed. Low-contrast linear defects are the weak spot;
higher resolution or a defect-specific supervised head would help.

**Checkpoint and data not in the repo.** The numbers above cannot be regenerated from
this repository alone: MVTec AD needs registration and the pretrained weights are
downloaded by torchvision. The tests use a small deterministic backbone, so they
verify configuration checking, seeding, calibration and serving, not detection quality.

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
src/data.py      MVTec loading, preprocessing config, fit/calibration split
src/train.py     Training, threshold calibration, evaluation, logged to MLflow
src/evaluate.py  Full metrics report, per-image CSV, box visualisations (new directory per run)
src/api.py       FastAPI serving; the checkpoint is loaded in the app lifespan
tests/           Checkpoint validation, seeding, calibration, API (run in CI)
reports/carpet/  The original recorded run, unchanged (see HISTORICAL.md); new runs go in runs/
Dockerfile       CPU-only image
```

---

## Reference

Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
Dataset: MVTec AD (Bergmann et al.), CC BY-NC-SA 4.0, non-commercial research use.

## Licence

The code is MIT (`LICENSE`). The MVTec-derived images in `reports/carpet/vis/` are CC BY-NC-SA 4.0 and the ImageNet-pretrained backbone weights have their own terms, so the pipeline is a research and education project. `THIRD_PARTY_NOTICES.md` has the details.
