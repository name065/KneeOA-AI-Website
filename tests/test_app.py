from io import BytesIO

import pytest
from PIL import Image

from app import app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "no-weights.keras"))
    app.config["TESTING"] = True
    with app.test_client() as browser:
        yield browser


def test_home(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"EfficientNet-B3" in res.data


def test_status_without_weights(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json["model_available"] is False
    assert len(response.json["classes"]) == 5


def test_missing_file(client):
    res = client.post("/api/predict", data={})
    assert res.status_code == 400


def test_invalid_file(client):
    res = client.post("/api/predict", data={"image": (BytesIO(b"not an image"), "fake.png")})
    assert res.status_code == 400


def test_missing_model_gives_503(client):
    buf = BytesIO()
    Image.new("RGB", (300, 300), "white").save(buf, "PNG")
    buf.seek(0)
    res = client.post("/api/predict", data={"image": (buf, "knee.png")})
    assert res.status_code == 503
    assert res.json["model_missing"] is True


def test_too_large(client):
    res = client.post("/api/predict", data=b"X" * (10 * 1024 * 1024 + 1024), content_type="application/octet-stream")
    assert res.status_code == 413
