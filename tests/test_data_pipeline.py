import pickle
from pathlib import Path

import numpy as np
import torch

from src.data_pipeline import MODULATION_CLASSES, SignalDataset, load_and_split_data


def test_normalization_logic(tmp_path: Path) -> None:
    # Arrange: Create a mock dataset dictionary containing raw non-normalized signals.
    # We use non-zero mean (+3.0) and non-unity standard deviation (*5.0) to test
    # whether load_and_split_data correctly rescales the signals.
    mock_data = {}
    classes_to_mock = ["QPSK", "BPSK"]
    snrs_to_mock = [2, 10]

    np.random.seed(42)
    for cls in classes_to_mock:
        for snr in snrs_to_mock:
            # Create raw samples of shape (num_samples, 2, 128)
            raw_samples = np.random.randn(100, 2, 128) * 5.0 + 3.0
            mock_data[(cls, snr)] = raw_samples

    pkl_path = tmp_path / "mock_dataset.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(mock_data, f)

    # Act: Load and preprocess data using our data pipeline function.
    train_split, val_split, test_split = load_and_split_data(
        str(pkl_path),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    x_train, _, _ = train_split
    x_val, _, _ = val_split
    x_test, _, _ = test_split

    # Assert: Verify that every sample in x_train, x_val, x_test is normalized
    # per-sample and per-channel independently to mean ~0 and std ~1.
    for split_name, x in [("train", x_train), ("val", x_val), ("test", x_test)]:
        # Compute mean and std for each channel of each sample along axis 2 (time).
        means = np.mean(x, axis=2)  # shape (N, 2)
        stds = np.std(x, axis=2)  # shape (N, 2)

        # Verify means are close to 0 and stds are close to 1 within a tolerance limit.
        assert np.allclose(means, 0.0, atol=1e-5), (
            f"Means not close to 0 in {split_name}"
        )
        assert np.allclose(stds, 1.0, atol=1e-5), f"Stds not close to 1 in {split_name}"


def test_stratified_splitting(tmp_path: Path) -> None:
    # Arrange: Create a mock dataset dictionary containing balanced class-SNR groups.
    # 3 classes, 3 SNRs. Each combination has 100 samples. Total samples = 900.
    mock_data = {}
    classes_to_mock = ["QPSK", "BPSK", "8PSK"]
    snrs_to_mock = [-10, 0, 10]

    for cls in classes_to_mock:
        for snr in snrs_to_mock:
            mock_data[(cls, snr)] = np.random.randn(100, 2, 128)

    pkl_path = tmp_path / "mock_dataset.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(mock_data, f)

    # Act: Split the dataset using 70/15/15 target ratios.
    train_split, val_split, test_split = load_and_split_data(
        str(pkl_path),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    x_train, y_train, snr_train = train_split
    x_val, y_val, snr_val = val_split
    x_test, y_test, snr_test = test_split

    # Assert total split counts:
    # 70% of 900 is 630.
    # 15% of 900 is 135.
    assert len(x_train) == 630
    assert len(x_val) == 135
    assert len(x_test) == 135

    # Assert stratification: for each class-SNR combination, the proportions
    # must be exactly 70%, 15%, 15% (yielding 70, 15, 15 samples).
    # This prevents class or noise imbalance across splits.
    for cls in classes_to_mock:
        cls_idx = MODULATION_CLASSES.index(cls)
        for snr in snrs_to_mock:
            # Train split counts
            train_mask = (y_train == cls_idx) & (snr_train == snr)
            assert np.sum(train_mask) == 70

            # Val split counts
            val_mask = (y_val == cls_idx) & (snr_val == snr)
            assert np.sum(val_mask) == 15

            # Test split counts
            test_mask = (y_test == cls_idx) & (snr_test == snr)
            assert np.sum(test_mask) == 15


def test_signal_dataset() -> None:
    # Arrange: Mock numpy data arrays
    x = np.random.randn(50, 2, 128).astype(np.float32)
    y = np.random.randint(0, 11, size=(50,)).astype(np.int64)

    # Act: Instantiate SignalDataset wrapping numpy inputs
    dataset = SignalDataset(x, y)

    # Assert: Verify dataset outputs correctly cast PyTorch Tensors on index retrieval.
    assert len(dataset) == 50
    x_sample, y_sample = dataset[0]

    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)
    assert x_sample.shape == (2, 128)
    assert y_sample.shape == ()
    assert x_sample.dtype == torch.float32
    assert y_sample.dtype == torch.long
    assert np.allclose(x_sample.numpy(), x[0])
    assert y_sample.item() == y[0]
