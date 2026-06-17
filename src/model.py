from typing import cast

import torch
from torch import nn


class SignalClassifier(nn.Module):
    """
    A 1D CNN architecture for RF signal modulation classification.
    Consists of 3 Conv1D blocks followed by global average pooling and a linear head.
    """

    def __init__(self, num_classes: int = 11) -> None:
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head
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


