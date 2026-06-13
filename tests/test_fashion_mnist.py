import torch

from fashion_mnist_classification.constants import CLASS_NAMES
from fashion_mnist_classification.model import FashionCNN


def test_fashion_cnn_output_shape():
    model = FashionCNN()
    batch = torch.randn(8, 1, 28, 28)
    output = model(batch)
    assert output.shape == (8, 10)


def test_class_names_count():
    assert len(CLASS_NAMES) == 10
