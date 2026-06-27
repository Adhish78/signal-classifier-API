from pathlib import Path

import numpy as np
import onnxruntime as ort


class InferenceEngine:
    """
    Wrapper around ONNX Runtime to handle RF signal classification inference.

    Rationale:
    Using PyTorch in a web-serving environment is resource-heavy and introduces
    unnecessary memory overhead. Compiling to ONNX and serving via ONNX Runtime
    provides a highly optimized, low-latency, and lightweight execution engine.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        # Initialize the ONNX session. We restrict execution to CPUExecutionProvider.
        # Rationale: API requests are served concurrently at low batch sizes
        # (typically 1 sample per request), where CPU execution provides excellent
        # throughput, low latency, and avoids GPU transfer overhead.
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

        # Retrieve input and output details dynamically to remain agnostic
        # to the model's exact signature keys.
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Run forward pass on (batch, 2, 128) input.
        Returns the predicted modulation class probabilities of shape (batch, 11).
        """
        if iq_data.dtype != np.float32:
            iq_data = iq_data.astype(np.float32)

        # Replicate identical z-score normalization along axis 2 (time steps)
        # to guarantee the input data matches the exact math distribution
        # the model was trained on.
        means = np.mean(iq_data, axis=2, keepdims=True)
        stds = np.std(iq_data, axis=2, keepdims=True)
        eps = 1e-10
        iq_data_normalized = (iq_data - means) / (stds + eps)

        # Execute inference on the ONNX Runtime engine.
        outputs = self.session.run(
            [self.output_name], {self.input_name: iq_data_normalized}
        )

        # Cast/verify type for static analysis
        result: np.ndarray = outputs[0]
        return result
