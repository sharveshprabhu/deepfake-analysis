import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert "endpoints" in data


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["database_connected"] is True
    assert "visual_ai" in data["ai_modules"]


def test_upload_and_tamper_detection():
    # 1. Upload valid image
    demo_file = Path("demo_media/sample_real.jpg")
    with open(demo_file, "rb") as f:
        res = client.post("/upload", files={"file": ("sample_real.jpg", f, "image/jpeg")})
    assert res.status_code == 201
    data = res.json()
    evidence_id = data["evidence_id"]

    # 2. Query Evidence Passport
    ev_res = client.get(f"/evidence/{evidence_id}")
    assert ev_res.status_code == 200
    ev_data = ev_res.json()
    assert ev_data["evidence_id"] == evidence_id
    assert ev_data["is_tampered"] is False

    # 3. List all evidence
    list_res = client.get("/evidence")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1


def test_ui_endpoint():
    res = client.get("/ui")
    assert res.status_code == 200
    assert "NEURALX" in res.text
    assert "TruthLens" in res.text


def test_diagnostics_frequency_srm_endpoint():
    res = client.get("/diagnostics/frequency-srm")
    assert res.status_code == 200
    data = res.json()
    assert "key_findings" in data
    assert "sample_audits" in data
    assert data["total_samples_audited"] > 0

