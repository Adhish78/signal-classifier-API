from pathlib import Path

import numpy as np
import pytest

from src.inference import InferenceEngine


def test_inference_engine_invalid_path() -> None:
    # Act & Assert: Initializing the InferenceEngine wrapper with a non-existent
    # path must fail-fast by raising a FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        InferenceEngine("non_existent_model_file.onnx")


def test_inference_engine_valid_inference() -> None:
    # Arrange: Locate the small mock ONNX model fixture compiled for testing.
    fixture_path = Path("tests") / "fixtures" / "test_model.onnx"
    assert fixture_path.exists(), "Test model fixture must exist"

    # Initialize the ONNX session wrapper engine
    engine = InferenceEngine(str(fixture_path))

    # Prepare input signals of shape (batch, 2, 128)
    batch_size = 4
    dummy_input = np.random.randn(batch_size, 2, 128).astype(np.float32)

    # Act: Evaluate the batch through the ONNX Runtime session.
    output = engine.predict(dummy_input)

    # Assert: Verify output shape and data properties.
    assert isinstance(output, np.ndarray)
    assert output.shape == (batch_size, 11)

    # Verify that the output values are probability distributions.
    # The outputs in each row must sum to exactly 1.0 (within float tolerances),
    # verifying that the Softmax activation wrapper is fully embedded in the ONNX graph.
    for i in range(batch_size):
        row_sum = np.sum(output[i])
        assert np.isclose(row_sum, 1.0, rtol=1e-5)
