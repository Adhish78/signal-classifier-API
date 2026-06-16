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
    output = engine.predict(iq_data_batched)
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
