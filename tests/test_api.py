"""API tests.

The app is started through its real lifespan (`api.app.router.lifespan_context`), because that is where the
checkpoint is loaded. Earlier versions of these tests never ran the lifespan, so `model` stayed None and the
prediction test passed by accepting a 503. Here a controlled checkpoint (tiny deterministic backbone, see
conftest.py) is served and /predict must succeed. The no-model case is its own test.

httpx 0.28 dropped the `app=` shortcut, and ASGITransport is async-only, so requests go through an async client.
"""

import httpx
import pytest
import pytest_asyncio
import torch

from src import api
from src.model import CheckpointError
from tests.conftest import SIZE, anomalous_image, normal_image, png_bytes


@pytest.fixture(autouse=True)
def isolate_api_state(monkeypatch):
    """The module keeps its model in globals; restore them so tests cannot see each other's state."""
    for name in ("MODEL_PATH", "INPUT_SIZE", "THRESHOLD"):
        monkeypatch.setattr(api, name, getattr(api, name))
    yield
    api.model = api.transform = api.threshold = api.threshold_source = None
    api.recent_scores.clear()


def serve(monkeypatch, path, input_size=None, threshold=None):
    monkeypatch.setattr(api, "MODEL_PATH", str(path))
    monkeypatch.setattr(api, "INPUT_SIZE", input_size)
    monkeypatch.setattr(api, "THRESHOLD", threshold)
    return api.app.router.lifespan_context(api.app)


def client_for():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")


@pytest_asyncio.fixture
async def served(checkpoint, monkeypatch):
    """The app started via its lifespan against the controlled checkpoint."""
    async with serve(monkeypatch, checkpoint["path"]):
        async with client_for() as c:
            c.checkpoint = checkpoint
            yield c


@pytest.mark.asyncio
async def test_lifespan_loads_the_checkpoint(served):
    body = (await served.get("/health")).json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["input_size"] == SIZE and body["seed"] == 0
    assert body["threshold"] == pytest.approx(served.checkpoint["threshold"]["value"])
    assert body["threshold_source"] == "checkpoint calibration"


@pytest.mark.asyncio
async def test_predict_scores_a_normal_image_below_threshold(served):
    r = await served.post("/predict", files={"file": ("n.png", png_bytes(normal_image(500)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"anomaly_score", "is_anomalous", "threshold", "latency_ms"}
    assert body["is_anomalous"] is False
    assert 0 <= body["anomaly_score"] <= body["threshold"]
    assert body["threshold"] == pytest.approx(served.checkpoint["threshold"]["value"])
    assert body["latency_ms"] > 0


@pytest.mark.asyncio
async def test_predict_flags_an_anomalous_image(served):
    r = await served.post("/predict", files={"file": ("a.png", png_bytes(anomalous_image()), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_anomalous"] is True
    assert body["anomaly_score"] > 5 * body["threshold"]


@pytest.mark.asyncio
async def test_api_score_matches_direct_model_call(served):
    """The service must report the number the model computes, not a transformed or stale one."""
    from tests.conftest import to_batch
    img = normal_image(501)
    expected = float(served.checkpoint["model"].predict(to_batch([img]))[0][0])
    r = await served.post("/predict", files={"file": ("n.png", png_bytes(img), "image/png")})
    assert r.json()["anomaly_score"] == pytest.approx(expected, rel=1e-4)


@pytest.mark.asyncio
async def test_predict_rejects_non_image_with_400(served):
    r = await served.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_stats_track_requests(served):
    assert (await served.get("/stats")).json()["n"] == 0
    for img in (normal_image(502), anomalous_image()):
        await served.post("/predict", files={"file": ("i.png", png_bytes(img), "image/png")})
    stats = (await served.get("/stats")).json()
    assert stats["n"] == 2 and stats["anomaly_rate"] == 0.5


@pytest.mark.asyncio
async def test_threshold_override_is_used_and_reported(checkpoint, monkeypatch):
    async with serve(monkeypatch, checkpoint["path"], threshold="123.5"):
        async with client_for() as c:
            health = (await c.get("/health")).json()
            assert health["threshold"] == 123.5 and health["threshold_source"] == "THRESHOLD environment variable"
            r = await c.post("/predict", files={"file": ("a.png", png_bytes(anomalous_image()), "image/png")})
            assert r.json()["is_anomalous"] is False        # an absurdly high threshold lets the defect through


@pytest.mark.asyncio
async def test_unavailable_model_gives_503_but_health_stays_up(tmp_path, monkeypatch):
    async with serve(monkeypatch, tmp_path / "missing.pt"):
        async with client_for() as c:
            health = (await c.get("/health")).json()
            assert health["status"] == "ok" and health["model_loaded"] is False
            r = await c.post("/predict", files={"file": ("n.png", png_bytes(normal_image(0)), "image/png")})
            assert r.status_code == 503


@pytest.mark.asyncio
async def test_restart_without_a_checkpoint_drops_the_old_model(checkpoint, tmp_path, monkeypatch):
    async with serve(monkeypatch, checkpoint["path"]):
        assert api.model is not None
    async with serve(monkeypatch, tmp_path / "missing.pt"):
        assert api.model is None and api.threshold is None


@pytest.mark.asyncio
async def test_startup_refuses_input_size_that_differs_from_the_checkpoint(checkpoint, monkeypatch):
    with pytest.raises(CheckpointError, match="requested input size 96"):
        async with serve(monkeypatch, checkpoint["path"], input_size="96"):
            pass


@pytest.mark.asyncio
async def test_startup_accepts_matching_input_size(checkpoint, monkeypatch):
    async with serve(monkeypatch, checkpoint["path"], input_size=str(SIZE)):
        assert api.model is not None


@pytest.mark.asyncio
async def test_startup_refuses_a_checkpoint_without_a_threshold(fitted, tmp_path, monkeypatch):
    model, _, _ = fitted
    path = tmp_path / "nothr.pt"
    model.save(str(path))                                  # no calibration recorded
    with pytest.raises(CheckpointError, match="no calibrated threshold"):
        async with serve(monkeypatch, path):
            pass
    async with serve(monkeypatch, path, threshold="0.5"):  # explicit threshold is accepted
        assert api.threshold == 0.5


@pytest.mark.asyncio
async def test_startup_refuses_a_tampered_checkpoint(checkpoint, monkeypatch):
    ckpt = torch.load(checkpoint["path"], weights_only=True)
    ckpt["config"]["preprocess"]["std"][0] = 9.0
    torch.save(ckpt, checkpoint["path"])
    with pytest.raises(CheckpointError, match="preprocessing"):
        async with serve(monkeypatch, checkpoint["path"]):
            pass
