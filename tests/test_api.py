"""API tests.

httpx 0.28 dropped the `app=` shortcut that starlette's TestClient used, and
ASGITransport is async-only -- so these drive the app through an async client.
The fixture is declared with pytest_asyncio.fixture so it works under strict
mode without needing a config file.

These run with or without a trained model, so CI stays green either way.
"""

import io

import httpx
import pytest
import pytest_asyncio
from PIL import Image

from src.api import app


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _png(size=(64, 64), colour=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_health_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@pytest.mark.asyncio
async def test_stats_shape(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    assert "n" in r.json()


@pytest.mark.asyncio
async def test_predict_rejects_non_image(client):
    from src import api

    if api.model is None:
        pytest.skip("no model loaded; /predict returns 503 before validation")

    r = await client.post(
        "/predict", files={"file": ("x.txt", b"not an image", "text/plain")}
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_predict_returns_expected_fields(client):
    from src import api

    r = await client.post("/predict", files={"file": ("x.png", _png(), "image/png")})

    if api.model is None:
        assert r.status_code == 503
        return

    assert r.status_code == 200
    body = r.json()
    for field in ("anomaly_score", "is_anomalous", "threshold", "latency_ms"):
        assert field in body
    assert isinstance(body["is_anomalous"], bool)
    assert body["anomaly_score"] >= 0