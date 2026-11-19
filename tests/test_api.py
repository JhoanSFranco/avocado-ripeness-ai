import pytest
from fastapi.testclient import TestClient

from avocado_ai.api.app import create_app
from avocado_ai.config import Settings
from tests.conftest import FakePredictor


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        feedback_path=tmp_path / "feedback.csv",
        model_path=tmp_path / "missing.pth",
        max_upload_mb=1,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, predictor=FakePredictor())) as test_client:
        yield test_client


def analyze(client, jpeg_bytes, session_id=None):
    data = {"session_id": session_id} if session_id else {}
    return client.post(
        "/api/v1/analyze", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}, data=data
    )


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info(client):
    body = client.get("/api/v1/model").json()
    assert [s["stage"] for s in body["stages"]] == [1, 2, 3, 4, 5]
    assert body["metrics"]["test_accuracy"] == 0.79


def test_analyze_returns_prediction_and_local_insight(client, jpeg_bytes):
    response = analyze(client, jpeg_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] and body["image_id"]
    assert body["prediction"]["stage"] == 3
    assert body["prediction"]["label"] == "Ripe"
    assert len(body["prediction"]["top_candidates"]) == 3
    assert body["insight"]["source"] == "local"


def test_same_session_fuses_photos(client, jpeg_bytes):
    analyze(client, jpeg_bytes, "avocado-1")
    body = analyze(client, jpeg_bytes, "avocado-1").json()
    assert body["prediction"]["images_used"] == 2
    assert body["image_prediction"]["images_used"] == 1


def test_clearing_a_session_restarts_fusion(client, jpeg_bytes):
    analyze(client, jpeg_bytes, "avocado-1")
    assert client.delete("/api/v1/sessions/avocado-1").status_code == 204
    assert analyze(client, jpeg_bytes, "avocado-1").json()["prediction"]["images_used"] == 1


def test_low_confidence_has_no_insight(settings, jpeg_bytes):
    uncertain = FakePredictor(probabilities=(0.3, 0.25, 0.2, 0.15, 0.1))
    with TestClient(create_app(settings, predictor=uncertain)) as client:
        body = analyze(client, jpeg_bytes).json()
    assert body["prediction"]["reliable"] is False
    assert body["insight"] is None


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ({"file": ("a.txt", b"hello", "text/plain")}, 415),
        ({"file": ("a.jpg", b"not really a jpeg", "image/jpeg")}, 400),
        ({"file": ("a.jpg", b"x" * (1024 * 1024 + 1), "image/jpeg")}, 413),
    ],
)
def test_invalid_uploads_are_rejected(client, files, status):
    assert client.post("/api/v1/analyze", files=files).status_code == status


def test_invalid_session_id_is_rejected(client, jpeg_bytes):
    assert analyze(client, jpeg_bytes, "../../etc").status_code == 422


def test_feedback_is_persisted(client, settings):
    payload = {
        "session_id": "s1",
        "image_id": "img1",
        "predicted_stage": 3,
        "is_correct": False,
        "correct_stage": 4,
    }
    assert client.post("/api/v1/feedback", json=payload).json() == {"total_feedback": 1}
    assert client.post("/api/v1/feedback", json=payload).json() == {"total_feedback": 2}
    assert "s1,img1,3,False,4" in settings.feedback_path.read_text()


def test_missing_model_degrades_gracefully(settings, jpeg_bytes):
    with TestClient(create_app(settings)) as client:  # model_path does not exist
        assert client.get("/api/v1/health").json()["status"] == "degraded"
        assert analyze(client, jpeg_bytes).status_code == 503
