import time

from fastapi import APIRouter, Request, Response, status

from api.config import Settings
from src.inference import InferenceEngine

router = APIRouter()


@router.get("/health")
def health_check(request: Request, response: Response) -> dict[str, str | bool | float]:
    # Calculate server uptime.
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time

    # Lazy-load the engine if not already set (common during test client sessions)
    if not hasattr(request.app.state, "inference_engine"):
        try:
            settings = Settings()
            request.app.state.inference_engine = InferenceEngine(settings.model_path)
        except Exception:
            pass

    inference_engine = getattr(request.app.state, "inference_engine", None)

    # Verify that the inference engine is loaded and contains an active session
    model_loaded = (
        inference_engine is not None
        and isinstance(inference_engine, InferenceEngine)
        and hasattr(inference_engine, "session")
    )

    # Rationale: Container orchestrators (e.g., Kubernetes liveness/readiness probes,
    # AWS ECS health checks) query this endpoint periodically. If the model file is
    # missing or the ONNX Runtime session fails to boot, returning HTTP 503 causes
    # the load balancer to stop routing client requests to this unhealthy container.
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "uptime": uptime,
            "detail": "ONNX Inference Engine is not initialized or loaded.",
        }

    return {"status": "healthy", "model_loaded": True, "uptime": uptime}
