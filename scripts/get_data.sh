#!/usr/bin/env bash
# MVTec AD is free for non-commercial research use.
# Register and download from:
#   https://www.mvtec.com/company/research/datasets/mvtec-ad
#
# Then extract one category into data/, e.g.:
#   mkdir -p data && tar -xf carpet.tar.xz -C data/
#
# Expected layout:
#   data/carpet/train/good/*.png
#   data/carpet/test/good/*.png
#   data/carpet/test/<defect>/*.png
set -euo pipefail
mkdir -p data artifacts
echo "Download MVTec AD manually, extract into ./data, then:"
echo "  python -m src.train --category carpet"
