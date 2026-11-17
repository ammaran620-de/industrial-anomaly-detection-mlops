import numpy as np
import pytest
import torch

from src.model import CheckpointError, PatchCore
from tests.conftest import SIZE, anomalous_image, normal_image, to_batch


def _tamper(path, edit):
    ckpt = torch.load(path, weights_only=True)
    edit(ckpt)
    torch.save(ckpt, path)


def _fresh(**kw):
    return PatchCore(input_size=kw.pop("input_size", SIZE), n_neighbors=3, device="cpu", pretrained=False, **kw)


def test_round_trip_restores_scores_config_and_threshold(checkpoint):
    original = checkpoint["model"]
    loaded = PatchCore.from_checkpoint(checkpoint["path"], device="cpu")
    batch = to_batch([normal_image(7), anomalous_image()])
    assert np.allclose(original.predict(batch)[0], loaded.predict(batch)[0])
    assert loaded.config()["input_size"] == SIZE and loaded.seed == 0
    assert loaded.threshold["value"] == checkpoint["threshold"]["value"]


def test_load_rejects_input_size_mismatch(checkpoint):
    with pytest.raises(CheckpointError, match="input_size"):
        _fresh(input_size=96).load(checkpoint["path"])


def test_from_checkpoint_rejects_requested_input_size_mismatch(checkpoint):
    with pytest.raises(CheckpointError, match="requested input size 96"):
        PatchCore.from_checkpoint(checkpoint["path"], input_size=96)
    PatchCore.from_checkpoint(checkpoint["path"], input_size=SIZE)  # the matching value is fine


def test_load_rejects_layer_mismatch(checkpoint):
    with pytest.raises(CheckpointError, match="layers"):
        PatchCore(layers=("layer2",), input_size=SIZE, device="cpu", pretrained=False).load(checkpoint["path"])


def test_load_rejects_backbone_weights_mismatch(checkpoint):
    with pytest.raises(CheckpointError, match="weights"):
        PatchCore(input_size=SIZE, device="cpu", pretrained=True).load(checkpoint["path"])


def test_load_rejects_preprocessing_mismatch(checkpoint):
    _tamper(checkpoint["path"], lambda c: c["config"]["preprocess"]["mean"].__setitem__(0, 0.5))
    with pytest.raises(CheckpointError, match="preprocessing"):
        _fresh().load(checkpoint["path"])


def test_load_rejects_wrong_feature_size(checkpoint):
    _tamper(checkpoint["path"], lambda c: c.update(memory_bank=c["memory_bank"][:, :10].contiguous()))
    with pytest.raises(CheckpointError, match="feature size"):
        _fresh().load(checkpoint["path"])


def test_load_rejects_backbone_that_produces_other_feature_size(checkpoint):
    def edit(c):
        c["memory_bank"] = torch.zeros(c["memory_bank"].shape[0], 99)
        c["config"]["feature_dim"] = 99     # internally consistent, but not what this backbone outputs
    _tamper(checkpoint["path"], edit)
    with pytest.raises(CheckpointError, match="24-dimensional"):
        _fresh().load(checkpoint["path"])


def test_load_rejects_bank_smaller_than_neighbours(checkpoint):
    _tamper(checkpoint["path"], lambda c: c.update(memory_bank=c["memory_bank"][:2].contiguous()))
    with pytest.raises(CheckpointError, match="fewer than n_neighbors"):
        _fresh().load(checkpoint["path"])


def test_load_rejects_feature_map_mismatch(checkpoint):
    _tamper(checkpoint["path"], lambda c: c["config"].update(feature_map_shape=[9, 9]))
    with pytest.raises(CheckpointError, match="feature map"):
        _fresh().load(checkpoint["path"])


def test_load_rejects_unknown_format_and_foreign_files(tmp_path, tiny_backbone):
    other = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, other)
    with pytest.raises(CheckpointError, match="not a PatchCore checkpoint"):
        _fresh().load(str(other))
    future = tmp_path / "future.pt"
    torch.save({"format": 99, "memory_bank": torch.zeros(4, 4)}, future)
    with pytest.raises(CheckpointError, match="unsupported checkpoint format"):
        _fresh().load(str(future))


def test_checkpoint_cannot_run_code_when_opened(tmp_path, tiny_backbone):
    """torch.load runs pickled code unless weights_only=True. A checkpoint is data, so this must fail closed."""
    class Evil:
        def __reduce__(self):
            return (print, ("code ran",))
    bad = tmp_path / "evil.pt"
    torch.save({"format": 2, "memory_bank": torch.zeros(4, 4), "payload": Evil()}, bad)
    with pytest.raises(CheckpointError, match="cannot read checkpoint"):
        _fresh().load(str(bad))


def test_legacy_checkpoint_loads_only_when_recorded_settings_match(fitted, tmp_path):
    model, _, _ = fitted
    legacy = {"memory_bank": model.memory_bank, "feature_map_shape": tuple(model.feature_map_shape),
              "layers": ("layer2", "layer3"), "input_size": SIZE, "n_neighbors": 3}   # the original format
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    loaded = PatchCore.from_checkpoint(str(path), device="cpu")
    assert loaded.threshold is None and loaded.seed is None
    assert np.allclose(loaded.predict(to_batch([anomalous_image()]))[0], model.predict(to_batch([anomalous_image()]))[0])
    with pytest.raises(CheckpointError, match="input_size"):
        _fresh(input_size=96).load(str(path))


def test_same_seed_gives_identical_bank_and_other_seed_differs(tiny_backbone):
    batches = [to_batch([normal_image(i) for i in range(6)])]
    def bank(seed):
        m = PatchCore(input_size=SIZE, coreset_ratio=0.2, n_neighbors=3, device="cpu", seed=seed, pretrained=False)
        return m.fit(batches).memory_bank
    assert torch.equal(bank(0), bank(0))
    assert not torch.equal(bank(0), bank(1))


def test_seed_also_controls_the_random_pool_step(tiny_backbone, monkeypatch):
    import src.model as m
    monkeypatch.setattr(m, "MAX_POOL", 500)          # force the random pool subsample to run
    batches = [to_batch([normal_image(i) for i in range(6)])]
    def bank(seed):
        return PatchCore(input_size=SIZE, coreset_ratio=0.2, n_neighbors=3, device="cpu", seed=seed,
                         pretrained=False).fit(batches).memory_bank
    assert bank(3).shape[0] == 100                   # 0.2 of the 500-patch pool, not of all 1,536 patches
    assert torch.equal(bank(3), bank(3)) and not torch.equal(bank(3), bank(4))


def test_fit_rejects_bank_smaller_than_neighbours(tiny_backbone):
    m = PatchCore(input_size=SIZE, coreset_ratio=0.0001, n_neighbors=9, device="cpu", seed=0, pretrained=False)
    with pytest.raises(ValueError, match="n_neighbors"):
        m.fit([to_batch([normal_image(0)])])


def test_real_wide_resnet_architecture_without_network():
    """No monkeypatching: the real backbone, random weights. Confirms the hooks and the 1536-d feature size."""
    m = PatchCore(input_size=SIZE, device="cpu", pretrained=False)
    out = m.extract(to_batch([normal_image(0)]))
    assert out.shape[1] == 1536 and m.feature_map_shape == (8, 8)
