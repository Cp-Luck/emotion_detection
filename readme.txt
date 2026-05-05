============================================================
  Emotion Detection from Face Images
  Machine Learning Final Project
============================================================

Student Name : Caleb Pham
CSM ID       : 10904096
Course       : Machine Learning
Language     : Python 3.10+

------------------------------------------------------------
PROJECT OVERVIEW
------------------------------------------------------------
This project trains a feedforward neural network to classify
48x48 grayscale face images into one of four emotions:
  - Happy
  - Sad
  - Angry
  - Neutral

It uses the FER-2013 dataset (Facial Expression Recognition).

------------------------------------------------------------
REQUIRED LIBRARIES
------------------------------------------------------------
  torch          - neural network framework (PyTorch)
  torchvision    - image utilities for PyTorch
  numpy          - numerical array operations
  pandas         - reading the CSV dataset
  matplotlib     - plotting sample predictions
  scikit-learn   - train/test split utility

All are listed in requirements.txt.

------------------------------------------------------------
PROJECT FILE STRUCTURE
------------------------------------------------------------
emotion_detection/
│
├── main.py                  ← Main script: load, train, test, predict
├── model.py                 ← Neural network class (EmotionNet)
├── requirements.txt         ← All required Python packages
├── readme.txt               ← This file
│
└── dataset/                 ← ⚠️ YOU MUST CREATE THIS FOLDER
    └── fer2013.csv          ← ⚠️ Place the dataset CSV here

After running, these files are created automatically:
  emotion_model.pth          ← Saved trained model weights
  sample_predictions.png     ← Grid of sample predictions

------------------------------------------------------------
WHERE TO PLACE THE DATASET
------------------------------------------------------------
1. Download fer2013.csv from Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013

   (You need a free Kaggle account. Click "Download" on the page.)

2. Inside the project folder, create a folder named 'dataset':
     emotion_detection/
     └── dataset/

3. Place the downloaded 'fer2013.csv' file inside that folder:
     emotion_detection/
     └── dataset/
         └── fer2013.csv        ← must be exactly this name

The code will throw a clear error message if the file is missing.

------------------------------------------------------------
HOW THE CODE IS STRUCTURED
------------------------------------------------------------
model.py
  - Defines EmotionNet, a 3-hidden-layer fully connected network
  - Input:  2304 neurons (48×48 pixels flattened)
  - Hidden: 512 → 256 → 128 neurons with ReLU + Dropout
  - Output: 4 neurons (one per emotion class)

main.py
  - load_fer2013()         Reads CSV, filters 4 emotions, parses pixels
  - preprocess()           Normalizes pixel values from 0–255 to 0–1
  - split_data()           80% train / 20% test split (stratified)
  - make_dataloaders()     Wraps data in PyTorch DataLoaders
  - train_model()          Trains with CrossEntropyLoss + Adam optimizer
  - evaluate_model()       Prints final test accuracy
  - show_sample_predictions() Plots 8 images with true/predicted labels
  - save_model()           Saves model weights to emotion_model.pth
  - main()                 Runs everything in order

------------------------------------------------------------
HOW TO RUN THE CODE  (step-by-step)
------------------------------------------------------------

STEP 1 — Make sure Python 3.10+ is installed
  python --version

STEP 2 — Open a terminal and navigate to the project folder
  cd path/to/emotion_detection

STEP 3 — (Recommended) Create a virtual environment
  python -m venv venv

  Activate it:
    Windows:   venv\Scripts\activate
    Mac/Linux: source venv/bin/activate

STEP 4 — Install required libraries
  pip install -r requirements.txt

STEP 5 — Place the dataset (see "WHERE TO PLACE THE DATASET" above)

STEP 6 — Run the main script
  python main.py

  Expected output:
    - Dataset loading info
    - Training progress (loss + accuracy per epoch)
    - Final test accuracy printed to console
    - A popup window with 8 sample predictions
    - 'emotion_model.pth' saved in the project folder
    - 'sample_predictions.png' saved in the project folder

------------------------------------------------------------
SUBMITTING TO CANVAS
------------------------------------------------------------
To create project.zip:

  Windows:
    - Right-click the 'emotion_detection' folder
    - Select "Send to" → "Compressed (zipped) folder"
    - Rename the zip to project.zip

  Mac:
    - Right-click 'emotion_detection'
    - Select "Compress 'emotion_detection'"
    - Rename to project.zip

  NOTE: Do NOT include the 'dataset/' folder in your zip
  (fer2013.csv is large ~300MB). The grader will supply their own.
  If required to include it, zip separately and note in submission.

------------------------------------------------------------
NOTES
------------------------------------------------------------
- Training takes ~1–3 minutes on CPU depending on your machine.
- If you have an NVIDIA GPU, PyTorch will use it automatically.
- Expected test accuracy: approximately 55–70% (FER-2013 is a
  challenging dataset even for deep CNNs).
- The model uses Dropout to reduce overfitting.
============================================================
