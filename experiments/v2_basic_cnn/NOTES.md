# v2 — Basic CNN

Second iteration. Switched from the CSV pipeline to `ImageFolder`-based
loading (`dataset/fer2013/train|test/<class>/`) and from a feedforward
network to a small CNN — 3 conv layers, no batch norm, no augmentation —
and expanded from 4 emotion classes to the full 7 FER-2013 classes.

Superseded by the CNN at the repo root, which adds batch normalization,
progressive dropout, data augmentation, weight decay, and an LR
scheduler on top of this same basic structure.

Run from inside this folder: `python main.py`
