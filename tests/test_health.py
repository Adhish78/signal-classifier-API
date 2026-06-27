from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    # Act: Query the API health check endpoint
    response = client.get("/health")

    # Assert: Confirm HTTP 200 status and verify the payload attributes.
    # Uptime must be non-negative, and the overall status should report "healthy".
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime" in data
    assert isinstance(data["uptime"], (int, float))
    assert data["uptime"] >= 0
