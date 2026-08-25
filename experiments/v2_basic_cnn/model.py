"""
model.py
--------
Defines EmotionNet using PyTorch.
This version works with image tensors shaped like:
    (batch_size, 1, 48, 48)
It outputs 7 emotion classes by default.
"""

import torch
import torch.nn as nn


class EmotionNet(nn.Module):
    """A simple CNN for FER2013 emotion classification."""

    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 48x48 -> 24x24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 24x24 -> 12x12

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 12x12 -> 6x6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
