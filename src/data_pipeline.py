import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Canonical list of modulation classes
MODULATION_CLASSES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]


def load_and_split_data(
    pkl_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Loads raw RML2016.10a pickle data, normalizes IQ channels per-sample
    independently, and returns stratified splits (train, val, test)
    considering modulation classes and SNRs.

    Returns:
        train_split: (x_train, y_train, snr_train)
        val_split: (x_val, y_val, snr_val)
        test_split: (x_test, y_test, snr_test)
        - x: shape (N, 2, 128), float32 normalized array
        - y: shape (N,), int64 class labels
        - snr: shape (N,), float32 SNR values in dB
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    with Path(pkl_path).open("rb") as f:
        # Load dataset dictionary using latin1 encoding
        data = pickle.load(f, encoding="latin1")


    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    train_x_list, train_y_list, train_snr_list = [], [], []
    val_x_list, val_y_list, val_snr_list = [], [], []
    test_x_list, test_y_list, test_snr_list = [], [], []

    # Sort keys to ensure deterministic processing order
    for key in sorted(data.keys()):
        cls_name, snr_val = key
        if cls_name not in MODULATION_CLASSES:
            continue

        class_idx = MODULATION_CLASSES.index(cls_name)
        samples = data[key]  # shape (num_samples, 2, 128)
        num_samples = len(samples)

        # Shuffle indices for this class-SNR group
        shuffled_indices = rng.permutation(num_samples)

        # Calculate split sizes
        n_train = round(num_samples * train_ratio)
        n_val = round(num_samples * val_ratio)

        train_idx = shuffled_indices[:n_train]
        val_idx = shuffled_indices[n_train : n_train + n_val]
        test_idx = shuffled_indices[n_train + n_val :]

        # Append train samples
        if len(train_idx) > 0:
            train_x_list.append(samples[train_idx])
            train_y_list.extend([class_idx] * len(train_idx))
            train_snr_list.extend([float(snr_val)] * len(train_idx))

        # Append val samples
        if len(val_idx) > 0:
            val_x_list.append(samples[val_idx])
            val_y_list.extend([class_idx] * len(val_idx))
            val_snr_list.extend([float(snr_val)] * len(val_idx))

        # Append test samples
        if len(test_idx) > 0:
            test_x_list.append(samples[test_idx])
            test_y_list.extend([class_idx] * len(test_idx))
            test_snr_list.extend([float(snr_val)] * len(test_idx))

    # Helper function to combine list and perform normalization
    def combine_and_normalize(
        x_list: list[np.ndarray], y_list: list[int], snr_list: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not x_list:
            return (
                np.empty((0, 2, 128), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        x = np.concatenate(x_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        snr = np.array(snr_list, dtype=np.float32)

        # Perform z-score normalization per-sample and per-channel independently.
        # x shape is (N, 2, 128).
        # We calculate mean and std along axis 2 (time step dimension).
        means = np.mean(x, axis=2, keepdims=True)  # (N, 2, 1)
        stds = np.std(x, axis=2, keepdims=True)  # (N, 2, 1)
        eps = 1e-10

        x_normalized = (x - means) / (stds + eps)
        return x_normalized, y, snr

    train_split = combine_and_normalize(train_x_list, train_y_list, train_snr_list)
    val_split = combine_and_normalize(val_x_list, val_y_list, val_snr_list)
    test_split = combine_and_normalize(test_x_list, test_y_list, test_snr_list)

    return train_split, val_split, test_split


class SignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    A PyTorch Dataset for wrapping signal samples and labels.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
