import argparse
import copy
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import mlflow
import torch
import torch.export
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data_pipeline import MODULATION_CLASSES, SignalDataset, load_and_split_data
from src.evaluate import evaluate_model
from src.model import SignalClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ONNXWrapper(nn.Module):
    """
    Wrapper PyTorch module to append a Softmax activation to the outputs
    of the SignalClassifier model specifically for serving/ONNX export.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return cast(torch.Tensor, self.softmax(logits))


def train_model(  # noqa: PLR0913, PLR0915
    data_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    output_dir: str,
    experiment_name: str = "Signal_Classifier",
    seed: int = 42,
) -> None:
    """
    Loads dataset, trains the model with MLflow metrics logging and early stopping,
    evaluates on the test set, and exports the model to ONNX alongside metadata.
    """
    logger.info("Setting up random seed: %d", seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set up MLflow tracking
    if "MLFLOW_ALLOW_FILE_STORE" not in os.environ:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        logger.info("Setting MLflow tracking URI to: %s", tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    logger.info("Setting MLflow experiment to: %s", experiment_name)
    mlflow.set_experiment(experiment_name)

    logger.info("Loading and splitting dataset from %s", data_path)
    train_split, val_split, test_split = load_and_split_data(
        pkl_path=data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )

    x_train, y_train, _ = train_split
    x_val, y_val, _ = val_split
    x_test, y_test, _ = test_split

    logger.info(
        "Split shapes - Train: %s, Val: %s, Test: %s",
        x_train.shape,
        x_val.shape,
        x_test.shape,
    )

    train_dataset = SignalDataset(x_train, y_train)
    val_dataset = SignalDataset(x_val, y_val)
    test_dataset = SignalDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = SignalClassifier(num_classes=len(MODULATION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    with mlflow.start_run() as run:
        logger.info("Started MLflow run: %s", run.info.run_id)

        # Log hyperparameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "early_stopping_patience": patience,
                "random_seed": seed,
                "device": str(device),
            }
        )

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in train_loader:
                inputs, targets = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(targets).sum().item()

            epoch_train_loss = train_loss / len(train_dataset)
            epoch_train_acc = correct_train / total_train

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    inputs, targets = batch_x.to(device), batch_y.to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total_val += targets.size(0)
                    correct_val += predicted.eq(targets).sum().item()

            epoch_val_loss = val_loss / len(val_dataset)
            epoch_val_acc = correct_val / total_val

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f, "
                "Val Loss: %.4f, Val Acc: %.4f",
                epoch,
                epochs,
                epoch_train_loss,
                epoch_train_acc,
                epoch_val_loss,
                epoch_val_acc,
            )

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_acc", epoch_val_acc, step=epoch)

            # Early stopping check
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                best_val_acc = epoch_val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                logger.info("Saved new best model checkpoint.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered. "
                        "Validation loss has not improved for %d epochs.",
                        patience,
                    )
                    break

        logger.info(
            "Training completed. Restoring best model weights from "
            "epoch %d (Val Loss: %.4f)",
            best_epoch,
            best_loss,
        )

        model.load_state_dict(best_model_weights)

        # Final evaluation on the test set
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                inputs, targets = batch_x.to(device), batch_y.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()

        final_test_loss = test_loss / len(test_dataset)
        final_test_acc = correct_test / total_test

        logger.info(
            "Test Set Evaluation - Loss: %.4f, Accuracy: %.4f",
            final_test_loss,
            final_test_acc,
        )

        # Log final test metrics and best parameters
        mlflow.log_metric("test_loss", final_test_loss)
        mlflow.log_metric("test_acc", final_test_acc)
        mlflow.log_metric("best_epoch", float(best_epoch))
        mlflow.log_param("best_val_loss", best_loss)
        mlflow.log_param("best_val_acc", best_val_acc)

        # Export best model checkpoint to ONNX format
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        onnx_file_path = output_path / "model.onnx"
        logger.info(
            "Exporting best model checkpoint to ONNX format at %s",
            onnx_file_path,
        )

        export_model = ONNXWrapper(model)
        export_model.to(device)
        export_model.eval()

        # Input shape: (batch_size, 2, 128)
        dummy_input = torch.randn(1, 2, 128, device=device)

        batch_dim = torch.export.Dim("batch_size", min=1)
        dynamic_shapes = {"x": {0: batch_dim}}

        torch.onnx.export(
            export_model,
            (dummy_input,),
            str(onnx_file_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )

        # Generate metadata.json
        metadata = {
            "model_version": "1.0.0",
            "framework": "ONNX",
            "classes": MODULATION_CLASSES,
            "input_shape": [2, 128],
            "training_accuracy": float(best_val_acc),
            "date_of_training": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        metadata_file_path = output_path / "metadata.json"
        logger.info("Generating metadata.json at %s", metadata_file_path)
        with metadata_file_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(onnx_file_path))
        mlflow.log_artifact(str(metadata_file_path))

        # Copy model.onnx to classifier.onnx for default API config compatibility
        classifier_file_path = output_path / "classifier.onnx"
        shutil.copy2(onnx_file_path, classifier_file_path)
        logger.info(
            "Copied model.onnx to %s for API compatibility",
            classifier_file_path,
        )

        # Trigger model evaluation reporting automatically
        logger.info("Automatically running evaluation pipeline on the test set...")
        try:
            reports_dir = Path("reports")
            evaluate_model(
                model_path=str(classifier_file_path),
                data_path=data_path,
                output_dir=str(reports_dir),
                experiment_name=experiment_name,
                run_id=run.info.run_id,
            )
            logger.info(
                "Evaluation pipeline completed successfully. "
                "Plots and metrics uploaded to MLflow."
            )
        except Exception as e:
            logger.exception("Failed to run automated evaluation pipeline: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Signal Classifier model and export to ONNX."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/RML2016.10a_dict.pkl",
        help="Path to the RML2016.10a_dict.pkl data file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping validation loss.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save exported ONNX model and metadata.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Signal_Classifier",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
