"""
main.py
-------
Emotion Detection from Face Images using an image-folder FER2013 dataset.

Expected dataset format:
    dataset/fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EmotionNet

# -----------------------------
# CONFIGURATION
# -----------------------------

DATASET_DIR = os.path.join("dataset", "fer2013")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = 48

# -----------------------------
# STEP 1: LOAD IMAGE DATASET
# -----------------------------

def make_dataloaders():
    """Loads FER2013 from train/test image folders using ImageFolder."""
    print("\n[1] Loading image-folder dataset")
    print(f"    Train folder: {TRAIN_DIR}")
    print(f"    Test folder : {TEST_DIR}")

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Train folder not found: {TRAIN_DIR}\n"
            "Expected format: dataset/fer2013/train/angry, happy, sad, etc."
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}\n"
            "Expected format: dataset/fer2013/test/angry, happy, sad, etc."
        )

    # Convert all images to grayscale, resize to 48x48, and convert to tensor.
    # Tensor values are automatically scaled from [0, 255] to [0.0, 1.0].
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

    print(f"    Classes found: {train_dataset.classes}")
    print(f"    Training images: {len(train_dataset)}")
    print(f"    Test images    : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

# -----------------------------
# STEP 2: TRAIN THE MODEL
# -----------------------------

def train_model(model, train_loader, device):
    print(f"\n[2] Training for {NUM_EPOCHS} epochs ...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        train_acc = 100.0 * correct / total
        print(f"    Epoch [{epoch:>2}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  Train Acc: {train_acc:.1f}%")

# -----------------------------
# STEP 3: EVALUATE THE MODEL
# -----------------------------

def evaluate_model(model, test_loader, device):
    print("\n[3] Evaluating on test set ...")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print("\n" + "=" * 40)
    print(f"  Final Test Accuracy: {accuracy:.2f}%")
    print("=" * 40 + "\n")
    return accuracy

# -----------------------------
# STEP 4: SHOW SAMPLE PREDICTIONS
# -----------------------------

def show_sample_predictions(model, test_loader, class_names, device, num_samples=8):
    print(f"[4] Saving {num_samples} sample predictions ...")

    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu()
    preds = preds.cpu()

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(14, 6))
    fig.suptitle("Sample Predictions", fontsize=13)

    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        color = "green" if labels[i].item() == preds[i].item() else "red"

        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=100)
    plt.show()
    print("    Saved plot to sample_predictions.png")

# -----------------------------
# STEP 5: SAVE THE MODEL
# -----------------------------

def save_model(model, path="emotion_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"[5] Model saved to {path}")

# -----------------------------
# MAIN
# -----------------------------

def main():
    print("=" * 50)
    print("  Emotion Detection from Face Images")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_loader, test_loader, class_names = make_dataloaders()

    model = EmotionNet(num_classes=len(class_names)).to(device)
    print(f"\nModel architecture:\n{model}")

    train_model(model, train_loader, device)
    evaluate_model(model, test_loader, device)
    show_sample_predictions(model, test_loader, class_names, device)
    save_model(model)

    print("\nDone!")

if __name__ == "__main__":
    main()
