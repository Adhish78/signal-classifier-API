import argparse
import os
from pathlib import Path

import matplotlib
import mlflow
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data_pipeline import MODULATION_CLASSES, load_and_split_data
from src.inference import InferenceEngine


def _compute_metrics(
    y_test: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """
    Computes summary classification metrics.
    """
    accuracy = float(np.mean(predictions == y_test))

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0.0
    )

    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average=None,
        labels=list(range(len(MODULATION_CLASSES))),
        zero_division=0.0,
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
    }

    for i, class_name in enumerate(MODULATION_CLASSES):
        metrics[f"class_{class_name}_precision"] = float(per_class_prec[i])
        metrics[f"class_{class_name}_recall"] = float(per_class_rec[i])
        metrics[f"class_{class_name}_f1"] = float(per_class_f1[i])

    return metrics


def _save_local_reports(
    y_test: np.ndarray,
    predictions: np.ndarray,
    snr_test: np.ndarray,
    out_path: Path,
) -> tuple[Path, Path, Path]:
    """
    Generates and saves the classification report, confusion matrix,
    and SNR-vs-accuracy plots locally.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Classification report text file
    report_text = classification_report(
        y_test, predictions, target_names=MODULATION_CLASSES, zero_division=0.0
    )
    report_file_path = out_path / "classification_report.txt"
    with report_file_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    # 2. Confusion matrix heatmap
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=MODULATION_CLASSES
    )
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    cm_path = out_path / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # 3. SNR-vs-accuracy curves
    unique_snrs = sorted(np.unique(snr_test))
    overall_accs = []
    per_class_accs: dict[str, list[float]] = {c: [] for c in MODULATION_CLASSES}

    for snr in unique_snrs:
        snr_mask = snr_test == snr
        if np.any(snr_mask):
            acc = np.mean(predictions[snr_mask] == y_test[snr_mask])
            overall_accs.append(acc)
        else:
            overall_accs.append(0.0)

        for i, class_name in enumerate(MODULATION_CLASSES):
            class_snr_mask = (snr_test == snr) & (y_test == i)
            if np.any(class_snr_mask):
                c_acc = np.mean(
                    predictions[class_snr_mask] == y_test[class_snr_mask]
                )
                per_class_accs[class_name].append(c_acc)
            else:
                per_class_accs[class_name].append(np.nan)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        unique_snrs,
        overall_accs,
        label="Overall",
        linewidth=3,
        marker="o",
        color="black",
    )
    for class_name, accs in per_class_accs.items():
        if not np.all(np.isnan(accs)):
            ax.plot(
                unique_snrs,
                accs,
                label=class_name,
                linestyle="--",
                marker="x",
                alpha=0.7,
            )

    ax.set_title("Accuracy vs SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    snr_path = out_path / "snr_vs_accuracy.png"
    fig.savefig(snr_path, dpi=150)
    plt.close(fig)

    return report_file_path, cm_path, snr_path


def _log_mlflow_run(  # noqa: PLR0913
    metrics: dict[str, float],
    report_file_path: Path,
    cm_path: Path,
    snr_path: Path,
    experiment_name: str,
    run_id: str | None,
) -> None:
    """
    Logs evaluation metrics and uploads artifacts to the designated MLflow run.
    """
    if "MLFLOW_ALLOW_FILE_STORE" not in os.environ:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Try auto-detecting the latest run ID if none provided
    if not run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["attribute.start_time DESC"],
                    max_results=1,
                )
                if runs:
                    run_id = runs[0].info.run_id
        except Exception:
            pass

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_file_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(snr_path))


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str,
    experiment_name: str = "Signal_Classifier",
    run_id: str | None = None,
) -> dict[str, float]:
    """
    Evaluates the trained ONNX model against the held-out test split.
    """
    # 1. Load the model using InferenceEngine
    engine = InferenceEngine(model_path)

    # 2. Load and split dataset
    _, _, test_split = load_and_split_data(
        pkl_path=data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    x_test, y_test, snr_test = test_split

    if len(x_test) == 0:
        raise ValueError("Test split is empty.")

    # 3. Run predictions on the test set
    probabilities = engine.predict(x_test)
    predictions = np.argmax(probabilities, axis=1)

    # 4. Compute metrics
    metrics = _compute_metrics(y_test, predictions)

    # 5. Generate and save reports and plots locally
    out_path = Path(output_dir)
    report_file, cm_file, snr_file = _save_local_reports(
        y_test, predictions, snr_test, out_path
    )

    # 6. Log metrics and artifacts to MLflow
    _log_mlflow_run(
        metrics=metrics,
        report_file_path=report_file,
        cm_path=cm_file,
        snr_path=snr_file,
        experiment_name=experiment_name,
        run_id=run_id,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Evaluation and Reporting for RF Signal Modulation "
            "Classifier Model."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.onnx",
        help="Path to the exported ONNX model file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/RML2016.10a_dict.pkl",
        help="Path to the RML2016.10a pickle data file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory where evaluation reports and plots will be saved.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Signal_Classifier",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional MLflow run ID to attach evaluation metrics "
            "and artifacts."
        ),
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
