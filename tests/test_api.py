"""Tests for the FastAPI backend layer using FastAPI's TestClient (no live server needed)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "InsightForge-AI API"


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body


def test_upload_and_analyze_flow(sample_dataframe, tmp_path):
    csv_path = tmp_path / "upload_test.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        upload_resp = client.post("/api/v1/upload", files={"file": ("upload_test.csv", f, "text/csv")})
    assert upload_resp.status_code == 200
    dataset_id = upload_resp.json()["dataset_id"]
    assert upload_resp.json()["n_rows"] == len(sample_dataframe)

    analyze_resp = client.post(f"/api/v1/datasets/{dataset_id}/analyze")
    assert analyze_resp.status_code == 200
    assert "kpis" in analyze_resp.json()

    quality_resp = client.get(f"/api/v1/datasets/{dataset_id}/quality")
    assert quality_resp.status_code == 200
    assert "health_score" in quality_resp.json()

    query_resp = client.post(
        f"/api/v1/datasets/{dataset_id}/query", json={"dataset_id": dataset_id, "question": "top 3 by revenue"}
    )
    assert query_resp.status_code == 200
    assert "answer" in query_resp.json()


def test_unknown_dataset_returns_404():
    response = client.get("/api/v1/datasets/does-not-exist/quality")
    assert response.status_code == 404
