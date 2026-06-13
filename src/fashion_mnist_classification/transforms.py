"""Training transforms used in the notebook."""

from __future__ import annotations

import PIL.Image
import torchvision.transforms as transforms


def get_train_transforms():
    """Return augmentation pipeline for Fashion-MNIST training."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(3, (0.1, 0.2)),
            transforms.RandomRotation(45, interpolation=PIL.Image.BILINEAR),
        ]
    )
