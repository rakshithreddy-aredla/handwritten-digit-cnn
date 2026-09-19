# Handwritten Digit Recognition (CNN) 🔢

A **Convolutional Neural Network** built with PyTorch that recognizes handwritten digits (0-9) from the MNIST dataset, reaching **~99% test accuracy**.

This is a deep learning project showing the complete pipeline: data loading, CNN architecture design, training loop, evaluation, and saving/loading a trained model.

## 🎯 Why this project matters

MNIST is the "hello world" of deep learning — but it's also a serious benchmark. A CNN here is the natural stepping stone to image recognition, self-driving cars, medical imaging, and more. Every interview panel recognizes this project.

## 🧠 CNN Architecture

```
Input: 28x28 grayscale image
  └─ Conv2D (32 filters, 3x3) → ReLU → MaxPool(2x2)
  └─ Conv2D (64 filters, 3x3) → ReLU → MaxPool(2x2)
  └─ Dropout (25%) to prevent overfitting
  └─ Flatten → Fully Connected (128) → ReLU
  └─ Fully Connected (10) → 10 digit classes
```

**Key ideas:**
- **Convolution** extracts local patterns (edges, curves) regardless of position
- **Pooling** shrinks the image while keeping important features
- **ReLU** adds non-linearity so the network can learn complex patterns
- **Dropout** randomly disables neurons during training to reduce overfitting

## 📊 Results

```
Epoch 1:  loss = 0.1456
Epoch 2:  loss = 0.0510
Epoch 3:  loss = 0.0356

Test accuracy: 0.9899 (98.99%)
```

Even at 99% accuracy, the model still confuses visually similar digits (e.g. 6↔0, 8↔9) — a great discussion point about the limits of image recognition.

## 📷 Outputs

- `predictions.png` - Grid of test digits with model predictions (green = correct, red = wrong)
- `digit_cnn.pth` - The trained model weights (ready to load for inference)

## 🚀 How to run

```bash
pip install -r requirements.txt
python digit_cnn.py
```

*First run downloads MNIST (~11MB) automatically.*

## 🏗️ Project Structure

```
05-handwritten-digit-cnn/
├── digit_cnn.py        # CNN model + training + evaluation
├── digit_cnn.pth       # Trained model weights
├── predictions.png     # Sample prediction visualization
├── requirements.txt
└── README.md
```

## 📚 Deep Learning Concepts Covered

- Convolutional Neural Networks (conv, pooling, dense layers)
- Training loop (forward pass, loss, backpropagation)
- Cross-entropy loss & Adam optimizer
- Overfitting prevention (dropout)
- Model saving/loading
- GPU vs CPU training (`DEVICE` detection)

## 💡 To Extend (great hackathon ideas)

- Build a **Flask web app** where users draw a digit and the model predicts it
- Increase accuracy with data augmentation or more epochs
- Try transfer learning on a larger dataset (e.g. Fashion-MNIST)
