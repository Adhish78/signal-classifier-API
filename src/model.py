from typing import cast

import torch
from torch import nn


class SignalClassifier(nn.Module):
    """
    A 1D CNN architecture for RF signal modulation classification.
    Consists of 3 Conv1D blocks followed by global average pooling and a linear head.
    
    Architectural Decisions:
    1. 1D Convolutions: Radio frequency (RF) IQ signals are 1D time-series data
       rather than 2D spatial images. 1D convolutions are used to extract local
       temporal relationships and correlation patterns across adjacent time-steps.
    2. Deep Feature Extraction: The channel capacity expands progressively
       (64 -> 128 -> 256) to capture increasingly complex abstract features
       from raw temporal sequences.
    3. Receptive Field Expansion: MaxPool1d halves the temporal dimension at each block,
       which forces subsequent convolutional layers to capture features across a wider
       receptive field.
    """

    def __init__(self, num_classes: int = 11) -> None:
        super().__init__()

        # Block 1: Captures fine-grained temporal micro-patterns from raw
        # IQ coordinates.
        # Input shape: (batch_size, 2, 128) -> Output shape: (batch_size, 64, 64)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 2: Captures intermediate temporal combinations.
        # Input shape: (batch_size, 64, 64) -> Output shape: (batch_size, 128, 32)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 3: Extracts high-level abstract representation patterns.
        # Input shape: (batch_size, 128, 32) -> Output shape: (batch_size, 256, 16)
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Global Average Pooling:
        # Instead of flattening the temporal sequence (which would lock the
        # network to a fixed input size and cause parameter explosion), we
        # collapse the remaining time dimension using AdaptiveAvgPool1d(1).
        # This outputs a fixed 256-dimensional feature vector regardless of
        # the input sequence length.
        # Input shape: (batch_size, 256, 16) -> Output shape: (batch_size, 256, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head:
        # Dropout (p=0.5) randomly zero-out features during training to
        # prevent co-adaptation, which is crucial for preventing overfitting
        # on noisy (low SNR) signals.
        self.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 2, 128)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Flatten (batch_size, 256, 1) to (batch_size, 256)
        return cast(torch.Tensor, self.head(x))
