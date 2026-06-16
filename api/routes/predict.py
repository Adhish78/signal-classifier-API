from typing import Any

from fastapi import APIRouter

from api.routes.metadata import MODULATION_CLASSES
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(_request: PredictionRequest) -> dict[str, Any]:
    # Generate dummy probabilities summing to exactly 1.0
    probabilities = dict.fromkeys(MODULATION_CLASSES, 0.0)
    probabilities["QPSK"] = 1.0

    return {
        "predicted_class": "QPSK",
        "probabilities": probabilities,
    }
