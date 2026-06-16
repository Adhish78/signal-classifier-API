import os
from pathlib import Path

# Configure the model path to point to the test fixture for all test runs
os.environ["SC_MODEL_PATH"] = str(Path("tests") / "fixtures" / "test_model.onnx")
