from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data import split_fit_calibration
from src.evaluate import choose_threshold, new_run_dir
from src.train import calibrate_threshold
from tests.conftest import anomalous_image, normal_image, to_batch


class Numbers(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i


def test_split_is_deterministic_disjoint_and_complete():
    a_fit, a_cal = split_fit_calibration(Numbers(100), 0.1, seed=7)
    b_fit, b_cal = split_fit_calibration(Numbers(100), 0.1, seed=7)
    assert a_fit.indices == b_fit.indices and a_cal.indices == b_cal.indices
    assert len(a_cal) == 10 and len(a_fit) == 90
    assert set(a_fit.indices).isdisjoint(a_cal.indices)
    assert sorted(a_fit.indices + a_cal.indices) == list(range(100))


def test_split_changes_with_seed_and_handles_edges():
    assert split_fit_calibration(Numbers(100), 0.1, 1)[1].indices != split_fit_calibration(Numbers(100), 0.1, 2)[1].indices
    fit, cal = split_fit_calibration(Numbers(20), 0.0, 0)
    assert cal is None and len(fit) == 20
    with pytest.raises(ValueError):
        split_fit_calibration(Numbers(20), 1.0, 0)
    with pytest.raises(ValueError):
        split_fit_calibration(Numbers(2), 0.9, 0)       # would leave nothing to fit on


def test_calibrated_threshold_is_the_percentile_of_the_heldout_scores(fitted):
    model, _, cal_imgs = fitted
    loader = [to_batch(cal_imgs[:4]), to_batch(cal_imgs[4:])]
    thr = calibrate_threshold(model, loader, percentile=90.0)
    expected = float(np.percentile(model.predict(to_batch(cal_imgs))[0], 90.0))
    assert thr["value"] == pytest.approx(expected)
    assert thr["n_calibration"] == len(cal_imgs) and thr["percentile"] == 90.0


def test_threshold_does_not_depend_on_defective_images(fitted):
    """Calibration sees normals only, so adding a defect to the test set cannot move the operating point."""
    model, _, cal_imgs = fitted
    before = calibrate_threshold(model, [to_batch(cal_imgs)])["value"]
    _ = model.predict(to_batch([anomalous_image()]))        # scoring a defect leaves the threshold alone
    assert calibrate_threshold(model, [to_batch(cal_imgs)])["value"] == before
    assert model.predict(to_batch([anomalous_image()]))[0][0] > 5 * before


def test_threshold_source_priority_and_labelling():
    cal = {"value": 0.5, "percentile": 95.0, "n_calibration": 28}
    assert choose_threshold(0.9, cal, 0.1)[0] == 0.9
    value, why = choose_threshold(None, cal, 0.1)
    assert value == 0.5 and "independent of the test set" in why and "28" in why
    value, why = choose_threshold(None, None, 0.1)
    assert value == 0.1 and "NOT independent" in why


def test_new_runs_never_overwrite_earlier_results(tmp_path):
    historical = tmp_path / "carpet"
    historical.mkdir()
    (historical / "metrics.txt").write_text("original run")
    first = new_run_dir(str(tmp_path), "carpet", None, datetime(2026, 11, 17, 12, 0, 0, tzinfo=timezone.utc))
    assert first == historical / "runs" / "20261117T120000Z"
    with pytest.raises(FileExistsError):
        new_run_dir(str(tmp_path), "carpet", None, datetime(2026, 11, 17, 12, 0, 0, tzinfo=timezone.utc))
    named = new_run_dir(str(tmp_path), "carpet", "seed1")
    assert named.name == "seed1"
    assert (historical / "metrics.txt").read_text() == "original run"
