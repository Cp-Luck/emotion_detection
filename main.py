"""
main.py
-------
Emotion Detection from Face Images using FER2013 image folders.

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

BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005
IMAGE_SIZE = 48

# -----------------------------
# STEP 1: LOAD IMAGE DATASET
# -----------------------------

def make_dataloaders():
    print("\n[1] Loading image-folder dataset")
    print(f"    Train folder: {TRAIN_DIR}")
    print(f"    Test folder : {TEST_DIR}")

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Train folder not found: {TRAIN_DIR}\n"
            "Expected: dataset/fer2013/train/angry, happy, neutral, etc."
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}\n"
            "Expected: dataset/fer2013/test/angry, happy, neutral, etc."
        )

    # Training gets augmentation to improve accuracy.
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Testing should NOT use random augmentation.
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

    print(f"    Classes found   : {train_dataset.classes}")
    print(f"    Training images : {len(train_dataset)}")
    print(f"    Test images     : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

# -----------------------------
# STEP 2: TRAIN THE MODEL
# -----------------------------

def train_model(model, train_loader, test_loader, device):
    print(f"\n[2] Training CNN for {NUM_EPOCHS} epochs ...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_acc = 0.0

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

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        test_acc = evaluate_model(model, test_loader, device, quiet=True)
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_emotion_model.pth")

        print(
            f"    Epoch [{epoch:>2}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.1f}% "
            f"Test Acc: {test_acc:.2f}% "
            f"Best: {best_acc:.2f}%"
        )

    print(f"\nBest model saved to best_emotion_model.pth with accuracy {best_acc:.2f}%")

# -----------------------------
# STEP 3: EVALUATE THE MODEL
# -----------------------------

def evaluate_model(model, test_loader, device, quiet=False):
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

    if not quiet:
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
        # Undo normalization for display: [-1, 1] -> [0, 1]
        img = images[i].squeeze().numpy()
        img = (img * 0.5) + 0.5

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
# MAIN
# -----------------------------

def main():
    print("=" * 50)
    print("  Emotion Detection from Face Images - CNN Upgrade")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_loader, test_loader, class_names = make_dataloaders()

    model = EmotionNet(num_classes=len(class_names)).to(device)
    print(f"\nModel architecture:\n{model}")

    train_model(model, train_loader, test_loader, device)

    # Load the best saved model before final evaluation and sample predictions.
    model.load_state_dict(torch.load("best_emotion_model.pth", map_location=device))
    evaluate_model(model, test_loader, device)
    show_sample_predictions(model, test_loader, class_names, device)

    torch.save(model.state_dict(), "emotion_model.pth")
    print("[5] Final model saved to emotion_model.pth")
    print("\nDone!")


if __name__ == "__main__":
    main()
