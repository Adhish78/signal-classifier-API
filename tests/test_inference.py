from pathlib import Path

import numpy as np
import pytest

from src.inference import InferenceEngine


def test_inference_engine_invalid_path() -> None:
    # Initializing with non-existent path should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        InferenceEngine("non_existent_model_file.onnx")


def test_inference_engine_valid_inference() -> None:
    fixture_path = Path("tests") / "fixtures" / "test_model.onnx"
    assert fixture_path.exists(), "Test model fixture must exist"

    # Initialize engine
    engine = InferenceEngine(str(fixture_path))

    # Prepare inputs of shape (batch, 2, 128)
    batch_size = 4
    dummy_input = np.random.randn(batch_size, 2, 128).astype(np.float32)

    # Run prediction
    output = engine.predict(dummy_input)

    # Validate output shape and properties
    assert isinstance(output, np.ndarray)
    assert output.shape == (batch_size, 11)

    # Assert that predictions sum to ~1.0
    for i in range(batch_size):
        row_sum = np.sum(output[i])
        assert np.isclose(row_sum, 1.0, rtol=1e-5)
