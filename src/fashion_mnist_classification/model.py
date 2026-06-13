"""CNN architecture for Fashion-MNIST classification."""

from __future__ import annotations

import torch.nn as nn


class FashionCNN(nn.Module):
    """Two-block CNN with batch normalization for 10-class Fashion-MNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear = nn.Sequential(nn.Linear(4 * 7 * 7, num_classes))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
