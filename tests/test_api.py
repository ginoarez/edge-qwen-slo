from fastapi.testclient import TestClient

from app.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_exposed():
    c = TestClient(app)
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "edge_qwen_requests_total" in r.text


def test_dashboard():
    c = TestClient(app)
    r = c.get("/dashboard")
    assert r.status_code == 200
    assert "edge-qwen-slo" in r.text
