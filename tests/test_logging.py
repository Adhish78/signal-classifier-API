import json
import logging
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_json_logging_middleware(caplog) -> None:
    # Set logging level to capture INFO messages
    caplog.set_level(logging.INFO)

    # Perform a request to health check
    response = client.get("/health")
    assert response.status_code == 200

    # Parse JSON logs emitted by the middleware
    json_logs = []
    for record in caplog.records:
        try:
            log_data = json.loads(record.message)
            if "path" in log_data and log_data["path"] == "/health":
                json_logs.append(log_data)
        except json.JSONDecodeError:
            continue

    # Assert exactly one log was output by the middleware for the request
    assert len(json_logs) == 1
    log = json_logs[0]
    assert log["method"] == "GET"
    assert log["path"] == "/health"
    assert log["status_code"] == 200
    assert "latency_ms" in log
    assert isinstance(log["latency_ms"], float)
    assert log["latency_ms"] >= 0.0
    assert "timestamp" in log
