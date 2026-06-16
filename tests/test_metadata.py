from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_metadata_endpoint() -> None:
    response = client.get("/model/metadata")
    assert response.status_code == 200

    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert data["framework"] == "ONNX"
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) == 11
    assert "QPSK" in data["classes"]
    assert data["input_shape"] == [2, 128]
    assert isinstance(data["training_accuracy"], float)
    assert isinstance(data["date_of_training"], str)
