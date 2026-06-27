import contextlib
import os
import pickle
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pytest
import torch
from torch import nn

from src.data_pipeline import MODULATION_CLASSES
from src.evaluate import evaluate_model
from src.evaluate import main as cli_main
from src.inference import InferenceEngine


@contextlib.contextmanager
def mock_start_run(*_args: Any, **_kwargs: Any) -> Generator[Any, None, None]:
    class MockRun:
        info = type("info", (), {"run_id": "mock_run_id"})()

    yield MockRun()


def create_mock_dataset(pkl_path: Path) -> None:
    # 11 classes, 2 SNRs, 10 samples each
    snrs = [-10, 10]
    mock_data = {}
    np.random.seed(42)
    for cls in MODULATION_CLASSES:
        for snr in snrs:
            mock_data[(cls, snr)] = np.random.randn(10, 2, 128).astype(np.float32)
    with pkl_path.open("wb") as f:
        pickle.dump(mock_data, f)


def create_mock_onnx_model(model_path: Path) -> None:
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2 * 128, len(MODULATION_CLASSES))
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_flat = x.view(x.size(0), -1)
            return cast(torch.Tensor, self.softmax(self.linear(x_flat)))

    model = DummyModel()
    model.eval()  # Avoid warnings by exporting in eval mode
    dummy_input = torch.randn(1, 2, 128)

    # Use dynamic_shapes instead of dynamic_axes
    batch_dim = torch.export.Dim("batch_size", min=1)
    dynamic_shapes = {"x": {0: batch_dim}}

    torch.onnx.export(
        model,
        (dummy_input,),
        str(model_path),
        input_names=["x"],
        output_names=["output"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        dynamo=True,
    )


def test_evaluate_pipeline_basic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: Set up mock model and data files.
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)

    model_path = tmp_path / "model.onnx"
    create_mock_onnx_model(model_path)

    output_dir = tmp_path / "reports"

    # Mock MLflow dependencies to avoid network connections during test runtime.
    monkeypatch.setattr("mlflow.start_run", mock_start_run)
    monkeypatch.setattr("mlflow.log_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_artifact", lambda *_args, **_kwargs: None)

    # Act: Trigger model evaluation.
    metrics = evaluate_model(
        model_path=str(model_path),
        data_path=str(pkl_path),
        output_dir=str(output_dir),
    )

    # Assert: Confirm basic accuracy limits.
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_metrics_calculation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: Setup mock model and dataset files.
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)

    model_path = tmp_path / "model.onnx"
    create_mock_onnx_model(model_path)

    output_dir = tmp_path / "reports"

    # Mock MLflow
    monkeypatch.setattr("mlflow.start_run", mock_start_run)
    monkeypatch.setattr("mlflow.log_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_artifact", lambda *_args, **_kwargs: None)

    # Setup target mock predictions for the 22 test samples in sorted keys order.
    target_preds = [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]

    # Mock InferenceEngine.predict to yield deterministic target predictions.
    # Rationale: This isolates metrics calculations from model inference behavior,
    # allowing us to test the exact arithmetic precision of macro and per-class
    # metric calculations against hand-computed expected outputs.
    def mock_predict(_self: Any, iq_data: np.ndarray) -> np.ndarray:
        out = np.zeros((len(iq_data), len(MODULATION_CLASSES)), dtype=np.float32)
        for i, p in enumerate(target_preds[: len(iq_data)]):
            out[i, p] = 1.0
        return out

    monkeypatch.setattr(InferenceEngine, "predict", mock_predict)

    # Act: Compute metrics.
    metrics = evaluate_model(
        model_path=str(model_path),
        data_path=str(pkl_path),
        output_dir=str(output_dir),
    )

    # Assert overall accuracy. Out of 22 samples, 20 are correct.
    assert pytest.approx(metrics["accuracy"]) == 20 / 22

    # Assert macro metrics.
    assert pytest.approx(metrics["macro_precision"]) == (9 + 4 / 3) / 11
    assert pytest.approx(metrics["macro_recall"]) == 10 / 11

    expected_macro_f1 = (7 + 4 / 3 + 1.6) / 11
    assert pytest.approx(metrics["macro_f1"]) == expected_macro_f1

    # Assert per-class metrics.
    assert pytest.approx(metrics["class_8PSK_f1"]) == 1.0
    assert pytest.approx(metrics["class_AM-DSB_f1"]) == 2 / 3
    assert pytest.approx(metrics["class_AM-SSB_f1"]) == 0.8
    assert pytest.approx(metrics["class_BPSK_f1"]) == 2 / 3
    assert pytest.approx(metrics["class_CPFSK_f1"]) == 0.8
    assert pytest.approx(metrics["class_GFSK_f1"]) == 1.0


def test_evaluate_local_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: Setup mock files.
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)

    model_path = tmp_path / "model.onnx"
    create_mock_onnx_model(model_path)

    output_dir = tmp_path / "reports"

    # Mock MLflow
    monkeypatch.setattr("mlflow.start_run", mock_start_run)
    monkeypatch.setattr("mlflow.log_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_artifact", lambda *_args, **_kwargs: None)

    # Act: Generate reports and plots.
    evaluate_model(
        model_path=str(model_path),
        data_path=str(pkl_path),
        output_dir=str(output_dir),
    )

    # Assert: Verify that the local text files and PNG plots exist and are non-empty.
    report_path = output_dir / "classification_report.txt"
    cm_path = output_dir / "confusion_matrix.png"
    snr_path = output_dir / "snr_vs_accuracy.png"

    assert report_path.exists()
    assert report_path.stat().st_size > 0
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0
    assert snr_path.exists()
    assert snr_path.stat().st_size > 0


def test_evaluate_mlflow_logging(tmp_path: Path) -> None:
    # Arrange: Setup mock data and local MLflow tracking
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)

    model_path = tmp_path / "model.onnx"
    create_mock_onnx_model(model_path)

    output_dir = tmp_path / "reports"

    # Configure MLflow local tracking
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlflow_tracking_dir.as_posix()}")

    # Create an initial training run to simulate logging to an existing run.
    mlflow.set_experiment("Test_Evaluation_MLflow")
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    # Act: run evaluate passing the run_id.
    evaluate_model(
        model_path=str(model_path),
        data_path=str(pkl_path),
        output_dir=str(output_dir),
        experiment_name="Test_Evaluation_MLflow",
        run_id=run_id,
    )

    # Assert: Verify that MLflow logged metrics and artifacts in the run.
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    # Verify overall and per-class metrics exist in the run
    metrics = run_data.data.metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "class_8PSK_f1" in metrics

    # Verify artifacts are logged
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "classification_report.txt" in artifacts
    assert "confusion_matrix.png" in artifacts
    assert "snr_vs_accuracy.png" in artifacts


def test_evaluate_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: Setup mock data, mock model, and command-line arguments.
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)

    model_path = tmp_path / "model.onnx"
    create_mock_onnx_model(model_path)

    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--model-path",
            str(model_path),
            "--data-path",
            str(pkl_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Mock MLflow
    monkeypatch.setattr("mlflow.start_run", mock_start_run)
    monkeypatch.setattr("mlflow.log_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlflow.log_artifact", lambda *_args, **_kwargs: None)

    # Act: Run evaluation via CLI main wrapper.
    cli_main()

    # Assert: Confirm CLI successfully ran the code and generated reports.
    assert (output_dir / "classification_report.txt").exists()
