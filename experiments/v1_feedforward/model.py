"""
model.py
--------
Defines the EmotionNet neural network using PyTorch.
The model takes flattened 48x48 grayscale images as input
and outputs probabilities for 4 emotion classes.
"""

import torch
import torch.nn as nn


class EmotionNet(nn.Module):
    """
    A simple feedforward neural network for emotion classification.

    Architecture:
        Input  -> 2304 neurons  (48 x 48 pixels, flattened)
        Layer1 -> 512  neurons  (fully connected + ReLU + Dropout)
        Layer2 -> 256  neurons  (fully connected + ReLU + Dropout)
        Layer3 -> 128  neurons  (fully connected + ReLU)
        Output ->   4  neurons  (one per emotion class)
    """

    def __init__(self):
        super(EmotionNet, self).__init__()

        # Input size: 48 * 48 = 2304 pixels per image
        input_size = 48 * 48

        self.network = nn.Sequential(
            # --- Hidden Layer 1 ---
            nn.Linear(input_size, 512),   # fully connected layer
            nn.ReLU(),                     # activation function (non-linearity)
            nn.Dropout(0.3),               # randomly zero 30% of neurons to reduce overfitting

            # --- Hidden Layer 2 ---
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # --- Hidden Layer 3 ---
            nn.Linear(256, 128),
            nn.ReLU(),

            # --- Output Layer ---
            # 4 outputs: happy, sad, angry, neutral
            nn.Linear(128, 4)
            # Note: No softmax here because nn.CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        """
        Forward pass: takes a batch of images and returns raw scores (logits).
        x shape: (batch_size, 2304)
        """
        return self.network(x)
