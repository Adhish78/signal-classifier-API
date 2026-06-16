from pathlib import Path

import numpy as np
import onnxruntime as ort


class InferenceEngine:
    """
    Wrapper around ONNX Runtime to handle RF signal classification inference.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        # Initialize the ONNX session using CPU execution provider
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

        # Retrieve input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Run forward pass on (batch, 2, 128) input.
        Returns the predicted modulation class probabilities of shape (batch, 11).
        """
        if iq_data.dtype != np.float32:
            iq_data = iq_data.astype(np.float32)

        # Run inference using the ONNX Runtime session
        outputs = self.session.run([self.output_name], {self.input_name: iq_data})

        # Cast/verify type for static analysis
        result: np.ndarray = outputs[0]
        return result
