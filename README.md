# Fashion-MNIST Classification with CNNs

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-CNN-E63946?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Dataset-Fashion--MNIST-457B9D?style=flat-square" alt="Fashion-MNIST" />
  <img src="https://img.shields.io/badge/Classes-10-A8DADC?style=flat-square" alt="Classes" />
  <img src="https://img.shields.io/badge/License-MIT-1D3557?style=flat-square" alt="License" />
</p>

**Convolutional neural network** for classifying grayscale fashion images from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

> Bachelor-era deep learning project — PyTorch CNN training, augmentation, and evaluation in Jupyter.

---

## Highlights

- Custom 2-block CNN with `BatchNorm2d` and max pooling
- Data augmentation: flips, blur, and rotation
- Adam optimizer with cross-entropy loss
- Training loss curves and validation accuracy

---

## Structure

```
├── notebooks/
│   └── fashion_mnist_cnn.ipynb
├── outputs/               # Saved plots and checkpoints
├── src/
│   └── fashion_mnist_classification/
│       ├── constants.py   # Class label names
│       ├── model.py       # FashionCNN architecture
│       └── transforms.py  # Augmentation pipeline
├── tests/
│   └── test_fashion_mnist.py
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/noorcs39/FashionMNIST_Classification.git
cd FashionMNIST_Classification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Fashion-MNIST downloads automatically to `data/` on first notebook run.

---

## Usage

```bash
jupyter notebook notebooks/fashion_mnist_cnn.ipynb
```

**Run tests:**

```bash
pytest tests/ -v
```

---

## Model Architecture

| Layer | Details |
| --- | --- |
| Conv Block 1 | Conv2d(1→4) → BatchNorm → ReLU → MaxPool |
| Conv Block 2 | Conv2d(4→4) → BatchNorm → ReLU → MaxPool |
| Classifier | Linear(196 → 10) |

---

## Classes

T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot

---

## Author

**Noor Uddin**  
📧 [noor.cs2@yahoo.com](mailto:noor.cs2@yahoo.com)  
🐙 [github.com/noorcs39](https://github.com/noorcs39)

---

## License

[MIT License](LICENSE)
