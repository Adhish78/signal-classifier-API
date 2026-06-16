from typing import Any

from fastapi import APIRouter

from api.schemas import ModelMetadataResponse

router = APIRouter()

MODULATION_CLASSES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]


@router.get("/model/metadata", response_model=ModelMetadataResponse)
def get_model_metadata() -> dict[str, Any]:
    return {
        "model_version": "1.0.0",
        "framework": "ONNX",
        "classes": MODULATION_CLASSES,
        "input_shape": [2, 128],
        "training_accuracy": 0.762,
        "date_of_training": "2026-06-15T19:54:24Z",
    }
