import os
from pathlib import Path

# Pytest auto-discovers and executes this file before running any test modules.
# Rationale: Setting 'SC_MODEL_PATH' here overrides default environment configs,
# ensuring all API client fixtures and inference tests load the lightweight
# mock 'test_model.onnx' fixture instead of looking for the production model.
# This makes the entire test suite self-contained and runnable offline.
os.environ["SC_MODEL_PATH"] = str(Path("tests") / "fixtures" / "test_model.onnx")
