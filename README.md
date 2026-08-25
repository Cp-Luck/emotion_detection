# Facial Emotion Classifier

A CNN that classifies 48x48 grayscale face images into one of the 7 FER-2013
emotions (angry, disgust, fear, happy, neutral, sad, surprise), trained
with PyTorch.

## Problem

FER-2013 is a well-known but genuinely hard benchmark: labels are noisy
(collected via web search + crowd annotation, not a controlled lab
setting), classes are imbalanced (disgust has ~7x fewer examples than
happy), and several emotions are visually close (fear/sad/angry share a
lot of facial-muscle overlap). Published CNN results on the full 7-class
problem typically land in the 65-73% range — this isn't a task where
95%+ is realistic with a small model and no pretraining.

## Architecture

```
Input (1, 48, 48)
├─ Conv Block 1: 2x Conv2d(32) + BatchNorm + ReLU → MaxPool → Dropout2d(0.15)   [48x48 → 24x24]
├─ Conv Block 2: 2x Conv2d(64) + BatchNorm + ReLU → MaxPool → Dropout2d(0.20)   [24x24 → 12x12]
├─ Conv Block 3: 2x Conv2d(128) + BatchNorm + ReLU → MaxPool → Dropout2d(0.25)  [12x12 → 6x6]
└─ Classifier: Flatten → Linear(256) + BatchNorm + ReLU → Dropout(0.50) → Linear(7)
```

Training: Adam (lr 5e-4, weight decay 1e-4), `ReduceLROnPlateau` on test
accuracy, 40 epochs, checkpointing the best-accuracy epoch rather than
just the final one. Training data is augmented (random horizontal flip,
±10° rotation, small translation) — validation/test data is not.

Full definition in [`model.py`](model.py); training loop in [`main.py`](main.py).

## Setup

```bash
pip install -r requirements.txt
```

Expects the FER-2013 image-folder layout:

```
dataset/fer2013/
├── train/
│   ├── angry/  ├── disgust/  ├── fear/  ├── happy/
│   ├── neutral/  ├── sad/  └── surprise/
└── test/
    └── (same 7 class folders)
```

<!-- TODO: confirm/insert the exact dataset source you used — this folder
     layout matches the Kaggle "FER-2013" image dataset (msambare/fer2013),
     but that's inferred from the directory structure, not verified. -->

## Usage

```bash
python main.py
```

Trains for 40 epochs, evaluates on the held-out test set, saves a sample
predictions grid to `sample_predictions.png`, and writes weights to
`best_emotion_model.pth` (best epoch) and `emotion_model.pth` (final
epoch). Uses CUDA automatically if available, CPU otherwise.

## Results

Evaluated the saved `best_emotion_model.pth` against the FER-2013 test
set (7,178 images):

**66.02% overall accuracy** (4,739/7,178)

| Class | Accuracy | Test images |
|---|---|---|
| happy | 88.22% | 1,774 |
| surprise | 81.11% | 831 |
| neutral | 66.42% | 1,233 |
| angry | 61.59% | 958 |
| sad | 52.77% | 1,247 |
| disgust | 45.05% | 111 |
| fear | 37.40% | 1,024 |

The pattern matches what's documented about FER-2013 generally: happy
and surprise have the most distinctive expressions and the most training
data; fear is the weakest class, commonly confused with sad/angry even
by human annotators, and disgust suffers from having by far the fewest
examples (111 test images vs. 1,774 for happy).

## Project structure

```
emotion_detection/
├── model.py                    current CNN architecture
├── main.py                     current training/eval pipeline
├── requirements.txt
└── experiments/                earlier iterations, kept to show progression
    ├── v1_feedforward/           original: CSV-based, 4 classes, plain FC network
    └── v2_basic_cnn/              CNN without batch norm/augmentation, 7 classes
```

## Iteration history

1. **v1 — feedforward baseline** ([`experiments/v1_feedforward/`](experiments/v1_feedforward/)):
   CSV-based FER-2013 loading, filtered to 4 emotions, a plain
   fully-connected network. No convolution — the model saw each image as
   a flat vector of 2,304 pixel values.
2. **v2 — basic CNN** ([`experiments/v2_basic_cnn/`](experiments/v2_basic_cnn/)):
   Switched to image-folder loading and all 7 FER-2013 classes, and
   replaced the FC network with a small CNN. No batch norm, no
   augmentation, no LR scheduling yet.
3. **Current — regularized CNN** (this folder): added batch
   normalization, progressive dropout, data augmentation, weight decay,
   and `ReduceLROnPlateau` scheduling on top of v2's structure, plus
   checkpointing on best test accuracy instead of just saving whatever
   the final epoch happened to land on.

## Future improvements

- Transfer learning from a pretrained backbone (e.g. a small ResNet) —
  the biggest likely accuracy gain given FER-2013's small image size and
  limited training data
- Class-weighted loss or oversampling for `disgust`, given its ~7x
  under-representation relative to `happy`
- A confusion matrix instead of just per-class accuracy, to see exactly
  which classes `fear` gets confused with
