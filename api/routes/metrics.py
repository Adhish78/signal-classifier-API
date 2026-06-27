import time

from fastapi import APIRouter, Request

from api.schemas import PredictionMetrics

router = APIRouter()


@router.get("/metrics", response_model=PredictionMetrics)
def get_metrics(request: Request) -> dict[str, float | int]:
    # Rationale: This endpoint exposes the current server state metrics to allow
    # monitoring infrastructure (e.g. Prometheus scrapers, Datadog agents) to
    # query server metrics, latency profiles, and error ratios in real-time.
    app_state = request.app.state
    uptime = time.time() - getattr(app_state, "start_time", time.time())

    return {
        "uptime_seconds": uptime,
        "total_predictions": getattr(app_state, "total_predictions", 0),
        "failed_predictions": getattr(app_state, "failed_predictions", 0),
        "average_inference_time_ms": getattr(
            app_state, "average_inference_time_ms", 0.0
        ),
        "min_inference_time_ms": getattr(app_state, "min_inference_time_ms", 0.0),
        "max_inference_time_ms": getattr(app_state, "max_inference_time_ms", 0.0),
    }
