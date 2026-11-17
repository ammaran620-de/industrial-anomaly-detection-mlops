# Recorded run, carpet (kept as produced)

`metrics.txt`, `per_image.csv` and `vis/` in this directory are the output of the original evaluation run.
They are preserved unchanged and are the source of the figures in the README.

How that run differs from the current code:

- The threshold (2.1189) is the 95th percentile of the scores of the 28 good **test** images, the same images the
  run then scored. It is not independent of the test set, and about 5% of the good images (2 of 28) are flagged by
  construction. `best achievable F1 threshold` in `metrics.txt` also uses the test labels and is a reference only.
- The random patch-pool subsample was not seeded. The run cannot be reproduced exactly.
- The checkpoint stored the memory bank and a few fields, with no preprocessing, seed or threshold. The checkpoint
  itself is not committed.
- `per_image.csv` contains Windows-style paths (`data\carpet\test\...`), so the run was made on Windows.
- Configuration, from the README comparison table: coreset ratio 0.05, input size 320, 500-patch bank, CPU.

The current `src.train` and `src.evaluate` use a held-out calibration split of normal training images and a seed,
and write each run to `reports/<category>/runs/<run-name>/` without touching this directory. Do not compare the two
sets of numbers without keeping that difference in mind. No run with the current protocol has been recorded yet.
