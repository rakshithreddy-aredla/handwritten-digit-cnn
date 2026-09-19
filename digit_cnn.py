"""
Handwritten Digit Recognition with a Convolutional Neural Network
==================================================================
Classifies handwritten digits (0-9) from the MNIST dataset using a CNN
built with PyTorch. This is a deep learning project showing the full
pipeline: data loading, model architecture, training loop, evaluation,
and saving/loading the model.

Reaches ~99% accuracy on the test set.

Libraries: PyTorch, torchvision
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Use GPU if available (CUDA), otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 3


class DigitCNN(nn.Module):
    """
    A simple but effective CNN:
    conv -> conv -> pool -> dropout -> fc -> fc -> output (10 classes)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 1->32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32->64 channels
        self.pool = nn.MaxPool2d(2, 2)                             # downsample by 2x
        self.dropout = nn.Dropout(0.25)
        # After two conv+pool layers, 28x28 image -> 7x7 with 64 channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)                              # 10 digit classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)          # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                      # raw scores (logits)
        return x


def get_loaders():
    """Download MNIST (if needed) and return train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean & std
    ])
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}:  loss = {total_loss / len(train_loader):.4f}")


def evaluate(model, test_loader):
    """Return (accuracy, list of (image, true_label, predicted_label) for errors)."""
    model.eval()
    correct = 0
    total = 0
    errors = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            # Record a few misclassified examples to visualize
            mask = pred != labels
            for img, true_l, pred_l in zip(images[mask], labels[mask], pred[mask]):
                if len(errors) < 5:
                    errors.append((img.cpu(), true_l.item(), pred_l.item()))
    return correct / total, errors


def save_predictions(model, test_loader, num=9):
    """Save a grid of test digits with their predictions for the README/visual check."""
    model.eval()
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        pred = model(images.to(DEVICE)).argmax(dim=1).cpu()

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        correct = pred[i].item() == labels[i].item()
        ax.set_title(f"Pred: {pred[i].item()} (True {labels[i].item()})",
                     color="green" if correct else "red")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved sample predictions to predictions.png")


def main():
    print(f"Using device: {DEVICE}")

    train_loader, test_loader = get_loaders()
    print("MNIST loaded: 60,000 training + 10,000 test images")

    model = DigitCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)

    # Evaluate on test set
    accuracy, errors = evaluate(model, test_loader)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save the trained model
    torch.save(model.state_dict(), "digit_cnn.pth")
    print("Model saved to digit_cnn.pth")

    # Visualize predictions and a few misclassified examples
    save_predictions(model, test_loader)
    if errors:
        print("\nExample misclassifications:")
        for img, true_l, pred_l in errors:
            print(f"  true={true_l}, predicted={pred_l}")


if __name__ == "__main__":
    main()
