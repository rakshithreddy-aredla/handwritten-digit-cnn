# MNIST from scratch with a CNN

A convolutional network in PyTorch, trained to **98.99%** test accuracy.

```
28x28 grayscale
  └─ Conv2D(32, 3x3) → ReLU → MaxPool(2x2)
  └─ Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
  └─ Dropout(0.25)
  └─ Flatten → Dense(128) → ReLU
  └─ Dense(10) → 10 classes

Epoch 1  loss 0.1456
Epoch 2  loss 0.0510
Epoch 3  loss 0.0356
```

Two convolutions are enough for this dataset. MNIST digits are 28x28 and highly structured, so a second convolution layer captures combinations of edges that a single layer can't — and beyond that you're adding parameters without adding accuracy.

The remaining errors are the interesting part: the model confuses 6↔0, 8↔9, 3↔5 and 3↔8. These aren't noise. They share the same handwriting styles, and a human looking at a genuinely ambiguous sample would hesitate too. `predictions.png` shows the misclassified ones in red.

The script detects `DEVICE` and uses a GPU when one is available. First run downloads MNIST (~11MB).

```bash
pip install -r requirements.txt
python digit_cnn.py
```

## Files

```
digit_cnn.py        # model, training loop, evaluation
digit_cnn.pth       # trained weights
predictions.png     # test grid, green = correct, red = wrong
requirements.txt
```