import time

from fastapi import APIRouter, Request, Response, status

from api.config import Settings
from src.inference import InferenceEngine

router = APIRouter()


@router.get("/health")
def health_check(request: Request, response: Response) -> dict[str, str | bool | float]:
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

    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "uptime": uptime,
            "detail": "ONNX Inference Engine is not initialized or loaded.",
        }

    return {"status": "healthy", "model_loaded": True, "uptime": uptime}
