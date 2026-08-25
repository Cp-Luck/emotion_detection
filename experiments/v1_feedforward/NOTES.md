# v1 — Feedforward baseline

The first version of this project. Loads FER-2013 from the Kaggle CSV
(`dataset/fer2013.csv`, columns `emotion,pixels,Usage`), filters down to
4 emotion classes (angry/happy/sad/neutral), and trains a plain
fully-connected network (2304 → 512 → 256 → 128 → 4) with no
convolution, no augmentation.

Superseded by `v2_basic_cnn/`, then by the CNN at the repo root — kept
here as-is to show the actual progression rather than only the end
result. Needs `pandas` and `scikit-learn` in addition to the root
`requirements.txt` (used for CSV parsing and the train/test split).

Run from inside this folder: `python main.py`
