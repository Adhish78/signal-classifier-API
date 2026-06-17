import pickle
from pathlib import Path
import numpy as np
import pytest
import mlflow
import json

from src.train import train_model


def create_mock_dataset(pkl_path: Path) -> None:
    # Create a small mock dataset dictionary matching RML2016.10a format
    # 11 classes, 2 SNRs, 10 samples per combination
    classes = [
        "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
        "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
    ]
    snrs = [-10, 10]
    mock_data = {}
    
    np.random.seed(42)
    for cls in classes:
        for snr in snrs:
            # Each sample has shape (2, 128)
            mock_data[(cls, snr)] = np.random.randn(10, 2, 128).astype(np.float32)
            
    with open(pkl_path, "wb") as f:
        pickle.dump(mock_data, f)


def test_successful_training_run(tmp_path: Path) -> None:
    # Arrange
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)
    output_dir = tmp_path / "models"

    
    # Configure MLflow to use a temporary tracking directory for this test
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlflow_tracking_dir.as_posix()}")
    
    # Act
    train_model(
        data_path=str(pkl_path),
        epochs=2,
        batch_size=16,
        lr=0.01,
        patience=2,
        output_dir=str(output_dir),
        experiment_name="Test_Signal_Classifier",
        seed=42,
    )
    
    # Assert output files exist
    model_path = output_dir / "model.onnx"
    classifier_path = output_dir / "classifier.onnx"
    metadata_path = output_dir / "metadata.json"
    
    assert model_path.exists(), "model.onnx was not generated"
    assert classifier_path.exists(), "classifier.onnx was not generated"
    assert metadata_path.exists(), "metadata.json was not generated"
    
    # Verify metadata content
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    assert metadata["model_version"] == "1.0.0"
    assert metadata["framework"] == "ONNX"
    assert len(metadata["classes"]) == 11
    assert "QPSK" in metadata["classes"]
    assert metadata["input_shape"] == [2, 128]
    assert isinstance(metadata["training_accuracy"], float)
    assert "date_of_training" in metadata


def test_early_stopping(tmp_path: Path) -> None:
    # Arrange
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)
    output_dir = tmp_path / "models"
    
    # Configure MLflow
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlflow_tracking_dir.as_posix()}")
    
    # Act: Train with patience=2, epochs=10, and lr=0.0
    # Since lr=0.0, weights do not change, validation loss remains constant and triggers early stopping.
    train_model(
        data_path=str(pkl_path),
        epochs=10,
        batch_size=16,
        lr=0.0,
        patience=2,
        output_dir=str(output_dir),
        experiment_name="Test_Signal_Classifier_ES",
        seed=42,
    )

    
    # Assert
    # Verify that the MLflow run metrics history is less than 10 epochs
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Test_Signal_Classifier_ES")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0
    run_id = runs[0].info.run_id
    
    val_loss_history = client.get_metric_history(run_id, "val_loss")
    # Early stopping should have triggered, so epochs run must be < 10
    assert len(val_loss_history) < 10
    assert len(val_loss_history) >= 2  # Must run at least 2 epochs to evaluate patience


def test_mlflow_logging(tmp_path: Path) -> None:
    # Arrange
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)
    output_dir = tmp_path / "models"
    
    # Configure MLflow
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlflow_tracking_dir.as_posix()}")
    
    # Act
    train_model(
        data_path=str(pkl_path),
        epochs=2,
        batch_size=8,
        lr=0.005,
        patience=3,
        output_dir=str(output_dir),
        experiment_name="Test_MLflow_Metrics",
        seed=123,
    )
    
    # Assert: Query MLflow Client
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Test_MLflow_Metrics")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1, "Expected at least one training run"

    
    # Let's find the active training run
    run = runs[0]
    run_id = run.info.run_id
    
    # Verify params
    params = run.data.params
    assert params["learning_rate"] == "0.005"
    assert params["batch_size"] == "8"
    assert params["epochs"] == "2"
    assert params["early_stopping_patience"] == "3"
    assert params["random_seed"] == "123"
    assert "device" in params
    
    # Verify final run metrics
    metrics = run.data.metrics
    assert "test_loss" in metrics
    assert "test_acc" in metrics
    assert "best_epoch" in metrics
    assert "best_val_loss" in params  # Best loss logged as parameter in our train.py
    assert "best_val_acc" in params
    
    # Verify epoch metrics history
    train_loss_history = client.get_metric_history(run_id, "train_loss")
    train_acc_history = client.get_metric_history(run_id, "train_acc")
    val_loss_history = client.get_metric_history(run_id, "val_loss")
    val_acc_history = client.get_metric_history(run_id, "val_acc")
    
    assert len(train_loss_history) == 2
    assert len(train_acc_history) == 2
    assert len(val_loss_history) == 2
    assert len(val_acc_history) == 2
    
    # Verify steps are logged correctly
    assert train_loss_history[0].step == 1
    assert train_loss_history[1].step == 2


def test_onnx_model_properties(tmp_path: Path) -> None:
    # Arrange
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    pkl_path = tmp_path / "mock_dataset.pkl"
    create_mock_dataset(pkl_path)
    output_dir = tmp_path / "models"
    
    # Configure MLflow
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlflow_tracking_dir.as_posix()}")
    
    # Act
    train_model(
        data_path=str(pkl_path),
        epochs=1,
        batch_size=8,
        lr=0.01,
        patience=1,
        output_dir=str(output_dir),
        experiment_name="Test_ONNX_Export",
        seed=42,
    )
    
    # Assert: Load ONNX model and run inference
    import onnxruntime as ort
    model_path = output_dir / "model.onnx"
    assert model_path.exists()
    
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test batch size of 1
    input_data_1 = np.random.randn(1, 2, 128).astype(np.float32)
    output_1 = session.run([output_name], {input_name: input_data_1})[0]
    assert output_1.shape == (1, 11)
    assert np.allclose(np.sum(output_1, axis=1), 1.0, atol=1e-5)
    
    # Test batch size of 5 (dynamic batch check)
    input_data_5 = np.random.randn(5, 2, 128).astype(np.float32)
    output_5 = session.run([output_name], {input_name: input_data_5})[0]
    assert output_5.shape == (5, 11)
    for row in output_5:
        assert np.isclose(np.sum(row), 1.0, rtol=1e-5)



