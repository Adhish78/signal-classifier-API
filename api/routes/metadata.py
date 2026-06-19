import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.config import Settings
from api.schemas import ModelMetadataResponse

router = APIRouter()
logger = logging.getLogger(__name__)

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
    settings = Settings()
    metadata_path = Path(settings.model_path).parent / "metadata.json"

    if not metadata_path.exists():
        logger.error("Model metadata file not found at: %s", metadata_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Model metadata file not found. "
                "Ensure the model has been trained and exported."
            ),
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "model_version": data.get("model_version", "unknown"),
                "framework": data.get("framework", "unknown"),
                "classes": data.get("classes", MODULATION_CLASSES),
                "input_shape": data.get("input_shape", [2, 128]),
                "training_accuracy": float(data.get("training_accuracy", 0.0)),
                "date_of_training": data.get("date_of_training", "unknown"),
            }
    except Exception as e:
        logger.exception("Failed to read model metadata file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read model metadata file: {e}",
        ) from e
