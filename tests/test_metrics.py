from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import app

# Disable raising server exceptions so that TestClient catches RuntimeErrors
# during simulated failures and returns a 500 response, allowing the test
# to assert status codes and verify that metrics (like failed_predictions) update.
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_metrics() -> Generator[None, None, None]:
    # Rationale: App state metrics are held globally in the FastAPI app instance.
    # To isolate each test and prevent state pollution from previous test runs,
    # we automatically reset all counters to zero before running each test case.
    app.state.total_predictions = 0
    app.state.failed_predictions = 0
    app.state.average_inference_time_ms = 0.0
    app.state.min_inference_time_ms = 0.0
    app.state.max_inference_time_ms = 0.0
    yield


def test_metrics_initial_state() -> None:
    # Act: Query metrics endpoint at startup
    response = client.get("/metrics")
    assert response.status_code == 200

    # Assert: Confirm initial values are exactly zero.
    data = response.json()
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0

    assert data["total_predictions"] == 0
    assert data["failed_predictions"] == 0
    assert data["average_inference_time_ms"] == 0.0
    assert data["min_inference_time_ms"] == 0.0
    assert data["max_inference_time_ms"] == 0.0


def test_metrics_updated_on_prediction() -> None:
    # Arrange & Act: Send a successful prediction request.
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]
    predict_response = client.post("/predict", json={"iq_data": valid_iq_data})
    assert predict_response.status_code == 200

    # Act: Query the metrics endpoint.
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()

    # Assert: Verify that counters and latency values are incremented.
    assert metrics["total_predictions"] == 1
    assert metrics["failed_predictions"] == 0
    assert metrics["average_inference_time_ms"] > 0.0
    assert metrics["min_inference_time_ms"] > 0.0
    assert metrics["max_inference_time_ms"] > 0.0
    assert metrics["average_inference_time_ms"] == metrics["min_inference_time_ms"]
    assert metrics["average_inference_time_ms"] == metrics["max_inference_time_ms"]


def test_metrics_accumulation_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: Set up mock signals.
    valid_iq_data = [[0.1] * 128, [-0.1] * 128]

    # Mock api.routes.predict.perf_counter to return deterministic times:
    # Call 1: starts at 10.0, ends at 10.015 -> 15.0ms latency
    # Call 2: starts at 20.0, ends at 20.025 -> 25.0ms latency
    time_values = [10.0, 10.015, 20.0, 20.025]
    time_iter = iter(time_values)

    monkeypatch.setattr("api.routes.predict.perf_counter", lambda: next(time_iter))

    # 1. Act: First prediction (expected latency = 15ms)
    resp1 = client.post("/predict", json={"iq_data": valid_iq_data})
    assert resp1.status_code == 200
    
    # Assert: Verify initial metrics match 15.0ms latency
    m1 = client.get("/metrics").json()
    assert pytest.approx(m1["average_inference_time_ms"]) == 15.0
    assert pytest.approx(m1["min_inference_time_ms"]) == 15.0
    assert pytest.approx(m1["max_inference_time_ms"]) == 15.0

    # 2. Act: Second prediction (expected latency = 25ms)
    resp2 = client.post("/predict", json={"iq_data": valid_iq_data})
    assert resp2.status_code == 200
    
    # Assert: Verify rolling average equals 20.0ms, min=15.0ms, max=25.0ms
    m2 = client.get("/metrics").json()
    assert m2["total_predictions"] == 2
    assert m2["failed_predictions"] == 0
    assert pytest.approx(m2["average_inference_time_ms"]) == 20.0
    assert pytest.approx(m2["min_inference_time_ms"]) == 15.0
    assert pytest.approx(m2["max_inference_time_ms"]) == 25.0

    # 3. Act: Third prediction - simulated prediction exception.
    # Rationale: We verify that when an error occurs, the server increments
    # failed_predictions but completely excludes the failed execution time
    # from latency stats (so it stays avg=20.0, min=15.0, max=25.0).
    monkeypatch.setattr("api.routes.predict.perf_counter", lambda: 30.0)
    engine = app.state.inference_engine

    def mock_predict(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(engine, "predict", mock_predict)

    resp3 = client.post("/predict", json={"iq_data": valid_iq_data})
    assert resp3.status_code == 500

    # Assert: Confirm failure count incremented but latency statistics remain unchanged.
    m3 = client.get("/metrics").json()
    assert m3["total_predictions"] == 3
    assert m3["failed_predictions"] == 1
    assert pytest.approx(m3["average_inference_time_ms"]) == 20.0
    assert pytest.approx(m3["min_inference_time_ms"]) == 15.0
    assert pytest.approx(m3["max_inference_time_ms"]) == 25.0
