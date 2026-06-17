from time import perf_counter
from typing import Any

import numpy as np
from fastapi import APIRouter, Request

from api.config import Settings
from api.routes.metadata import MODULATION_CLASSES
from api.schemas import PredictionRequest, PredictionResponse
from src.inference import InferenceEngine

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, prediction_request: PredictionRequest) -> dict[str, Any]:
    # Convert input list to numpy array of shape (1, 2, 128)
    iq_data = np.array(prediction_request.iq_data, dtype=np.float32)
    iq_data_batched = np.expand_dims(iq_data, axis=0)

    # Retrieve engine from app state or initialize if not already set (e.g. tests)

    if not hasattr(request.app.state, "inference_engine"):
        settings = Settings()
        request.app.state.inference_engine = InferenceEngine(settings.model_path)

    engine = request.app.state.inference_engine
    
    app_state = request.app.state
    try:
        start_time = perf_counter()
        output = engine.predict(iq_data_batched)
        latency_ms = (perf_counter() - start_time) * 1000.0
    except Exception:
        if hasattr(app_state, "total_predictions"):
            app_state.total_predictions += 1
            app_state.failed_predictions += 1
        raise

    if hasattr(app_state, "total_predictions"):
        old_total = app_state.total_predictions
        old_failed = app_state.failed_predictions
        old_success = old_total - old_failed
        
        new_success = old_success + 1
        app_state.total_predictions += 1
        
        if old_success == 0:
            app_state.average_inference_time_ms = latency_ms
            app_state.min_inference_time_ms = latency_ms
            app_state.max_inference_time_ms = latency_ms
        else:
            old_avg = app_state.average_inference_time_ms
            app_state.average_inference_time_ms = old_avg + (latency_ms - old_avg) / new_success
            app_state.min_inference_time_ms = min(app_state.min_inference_time_ms, latency_ms)
            app_state.max_inference_time_ms = max(app_state.max_inference_time_ms, latency_ms)

    probabilities_list = output[0]

    # Map the probability outputs to modulation class labels
    probabilities = {
        cls: float(prob)
        for cls, prob in zip(MODULATION_CLASSES, probabilities_list, strict=True)
    }

    # Identify class with highest probability
    predicted_class = max(probabilities, key=lambda k: probabilities[k])

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
    }
