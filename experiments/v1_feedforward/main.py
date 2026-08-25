"""
main.py
-------
Emotion Detection from Face Images
-----------------------------------
This script:
  1. Loads and preprocesses the FER-2013 dataset
  2. Filters to 4 emotion classes: happy, sad, angry, neutral
  3. Trains a neural network (EmotionNet) defined in model.py
  4. Evaluates accuracy on the test set
  5. Shows sample predictions with true vs predicted labels
  6. Saves the trained model to 'emotion_model.pth'
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import EmotionNet

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Path to the FER-2013 CSV file.
# Place 'fer2013.csv' inside a folder called 'dataset/' next to this script.
DATASET_PATH = os.path.join("dataset", "fer2013.csv")

# The 4 emotions we care about (FER-2013 uses integer labels)
# FER-2013 label map: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
TARGET_EMOTIONS = {
    0: "angry",
    3: "happy",
    4: "sad",
    6: "neutral"
}

# Training settings
BATCH_SIZE  = 64      # number of images processed at once
NUM_EPOCHS  = 20      # how many full passes through the training data
LEARNING_RATE = 0.001 # step size for the optimizer
TEST_SPLIT  = 0.2     # 20% of data reserved for testing
RANDOM_SEED = 42      # for reproducibility

# ──────────────────────────────────────────────
# STEP 1: LOAD THE DATASET
# ──────────────────────────────────────────────

def load_fer2013(csv_path):
    """
    Reads the FER-2013 CSV and returns images (X) and labels (y).
    The CSV has columns: emotion, pixels, Usage
    Each 'pixels' entry is a space-separated string of 2304 integers.
    """
    print(f"\n[1] Loading dataset from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at '{csv_path}'.\n"
            "Please download fer2013.csv from Kaggle and place it in the 'dataset/' folder.\n"
            "See readme.txt for exact instructions.\n"
        )

    df = pd.read_csv(csv_path)
    print(f"    Total rows in CSV: {len(df)}")

    # Keep only the 4 target emotions
    df = df[df["emotion"].isin(TARGET_EMOTIONS.keys())].copy()
    print(f"    Rows after filtering to 4 emotions: {len(df)}")

    # Remap original FER labels → 0-based indices (0=angry,1=happy,2=sad,3=neutral)
    label_remap = {0: 0, 3: 1, 4: 2, 6: 3}
    df["label"] = df["emotion"].map(label_remap)

    # Parse pixel strings into numpy arrays
    # Each row's 'pixels' looks like "70 80 82 72 ..."
    X = np.array([
        list(map(int, row.split()))
        for row in df["pixels"]
    ], dtype=np.float32)

    y = df["label"].values.astype(np.int64)

    print(f"    Image array shape: {X.shape}  (samples × pixels)")
    print(f"    Label distribution: { {TARGET_EMOTIONS[k]: int((df['emotion']==k).sum()) for k in TARGET_EMOTIONS} }")
    return X, y


# ──────────────────────────────────────────────
# STEP 2: PREPROCESS
# ──────────────────────────────────────────────

def preprocess(X):
    """
    Normalizes pixel values from [0, 255] to [0.0, 1.0].
    Neural networks train better with small input values.
    """
    print("\n[2] Preprocessing: normalizing pixel values to [0, 1]")
    X = X / 255.0
    return X


# ──────────────────────────────────────────────
# STEP 3: SPLIT INTO TRAIN / TEST
# ──────────────────────────────────────────────

def split_data(X, y):
    """Splits data into training and test sets."""
    print(f"\n[3] Splitting data: {int((1-TEST_SPLIT)*100)}% train / {int(TEST_SPLIT*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y        # keeps class balance the same in both splits
    )
    print(f"    Training samples : {len(X_train)}")
    print(f"    Test samples     : {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# STEP 4: CREATE PYTORCH DATALOADERS
# ──────────────────────────────────────────────

def make_dataloaders(X_train, X_test, y_train, y_test):
    """
    Wraps numpy arrays in PyTorch Tensors and DataLoaders.
    DataLoaders handle batching and shuffling automatically.
    """
    print("\n[4] Creating PyTorch DataLoaders")

    # Convert numpy → torch tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t  = torch.tensor(X_test)
    y_test_t  = torch.tensor(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, X_test_t, y_test_t


# ──────────────────────────────────────────────
# STEP 5: TRAIN THE MODEL
# ──────────────────────────────────────────────

def train_model(model, train_loader, device):
    """
    Trains the neural network using:
      - CrossEntropyLoss (standard for multi-class classification)
      - Adam optimizer   (an adaptive gradient descent variant)
    Prints loss at the end of each epoch.
    """
    print(f"\n[5] Training for {NUM_EPOCHS} epochs ...")

    # Loss function: Cross-Entropy is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam adapts the learning rate automatically per parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()   # set model to training mode (enables Dropout)

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # --- Forward pass ---
            outputs = model(images)            # raw scores (logits)
            loss    = criterion(outputs, labels)  # compute loss

            # --- Backward pass (backpropagation) ---
            optimizer.zero_grad()   # clear old gradients
            loss.backward()         # compute new gradients
            optimizer.step()        # update weights

            # Track stats
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

        avg_loss = total_loss / total
        train_acc = 100.0 * correct / total
        print(f"    Epoch [{epoch:>2}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  Train Acc: {train_acc:.1f}%")


# ──────────────────────────────────────────────
# STEP 6: EVALUATE ON TEST SET
# ──────────────────────────────────────────────

def evaluate_model(model, test_loader, device):
    """
    Runs the model on the test set and prints final accuracy.
    No gradient computation needed during evaluation.
    """
    print("\n[6] Evaluating on test set ...")

    model.eval()    # disable Dropout during evaluation
    correct = 0
    total   = 0

    with torch.no_grad():   # no backprop needed for evaluation
        for images, labels in test_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"\n{'='*40}")
    print(f"  ✅ Final Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*40}\n")
    return accuracy


# ──────────────────────────────────────────────
# STEP 7: SHOW SAMPLE PREDICTIONS
# ──────────────────────────────────────────────

def show_sample_predictions(model, X_test_t, y_test_t, device, num_samples=8):
    """
    Plots a grid of sample images with true and predicted emotion labels.
    Green title = correct prediction, Red title = wrong prediction.
    """
    print(f"[7] Showing {num_samples} sample predictions ...")

    # Friendly names for the 4 remapped labels
    emotion_names = ["angry", "happy", "sad", "neutral"]

    model.eval()
    with torch.no_grad():
        outputs   = model(X_test_t[:num_samples].to(device))
        _, preds  = torch.max(outputs, 1)

    images = X_test_t[:num_samples].numpy()
    truths = y_test_t[:num_samples].numpy()
    preds  = preds.cpu().numpy()

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(14, 6))
    fig.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=13)

    for i, ax in enumerate(axes.flat):
        img = images[i].reshape(48, 48)   # restore 2D shape for display
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        true_label = emotion_names[truths[i]]
        pred_label = emotion_names[preds[i]]
        color = "green" if truths[i] == preds[i] else "red"

        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=100)
    plt.show()
    print("    Saved plot to 'sample_predictions.png'")


# ──────────────────────────────────────────────
# STEP 8: SAVE THE MODEL
# ──────────────────────────────────────────────

def save_model(model, path="emotion_model.pth"):
    """
    Saves only the learned weights (state_dict), not the entire model object.
    This is PyTorch's recommended way to save models.
    """
    torch.save(model.state_dict(), path)
    print(f"[8] Model saved to '{path}'")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Emotion Detection from Face Images")
    print("=" * 50)

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n    Using device: {device}")

    # Run pipeline
    X, y                              = load_fer2013(DATASET_PATH)
    X                                 = preprocess(X)
    X_train, X_test, y_train, y_test  = split_data(X, y)
    train_loader, test_loader, X_test_t, y_test_t = make_dataloaders(
                                            X_train, X_test, y_train, y_test)

    # Build model and move to device
    model = EmotionNet().to(device)
    print(f"\n    Model architecture:\n{model}")

    train_model(model, train_loader, device)
    evaluate_model(model, test_loader, device)
    show_sample_predictions(model, X_test_t, y_test_t, device)
    save_model(model)

    print("\nDone! ✅")


if __name__ == "__main__":
    main()
