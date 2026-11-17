# Third-party notices

`LICENSE` (MIT) covers the original code, tests, documentation and recorded figures in this repository. It does not cover the items below.

## Dataset images

`reports/carpet/vis/` holds visualisations that contain images from MVTec AD (Bergmann et al.). MVTec AD is published under CC BY-NC-SA 4.0: attribution, non-commercial use only, and share-alike. Those images are not MIT. The dataset itself is not in this repository; `scripts/get_data.sh` points to the registration page. This is the licence MVTec states for the dataset; I could not re-read it from this environment, so check it at mvtec.com before any reuse. Because of the non-commercial term, a model trained on MVTec AD is a research artefact here, not something to deploy commercially without checking.

## Pretrained backbone

The feature extractor is the torchvision `wide_resnet50_2` model with its `IMAGENET1K_V1` weights, downloaded at run time and not stored here. torchvision's code is BSD-3-Clause. The weights were trained on ImageNet, whose access terms are, as I understand them, limited to non-commercial research and education (not re-read from this environment). I have not resolved how that carries over to a PatchCore memory bank built from those features. Treat the whole pipeline as research and education use until that is checked.

## Method

The method is PatchCore (Roth et al., CVPR 2022). No third-party PatchCore source is vendored: the repository has no third-party directory and no file carries another party's copyright header.

## Python dependencies

PyTorch, torchvision, FastAPI, MLflow, scikit-learn, SciPy, matplotlib and the others in `requirements.txt` keep their own licences and are installed, not bundled.
