# Voice-Emotion-Detection
This project implements a machine‑learning pipeline to automatically recognize human emotions from short voice recordings. By extracting key acoustic features and training classification models, it aims to distinguish between several emotional states (e.g., neutral, calm, happy, sad, angry, fearful, disgust, and surprised) with high accuracy.


# Voice Emotion Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Project Overview

This repository implements an end‑to‑end **voice‑based emotion recognition** pipeline using classic ML and audio signal processing:

1. **Data Loading & Labeling**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature Extraction** (MFCC, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz)  
4. **Model Training & Evaluation** (MLP, Random Forest, SVM)  
5. **Interactive Chat Prototype** with emotion‑aware responses

---

voice-emotion-detection/
├── voice-emotion-detection.ipynb # Main Jupyter notebook
├── requirements.txt # Python dependencies
├── README.md # This file
├── LICENSE # MIT license
└── data/ # (optional) audio files organized by emotion
└── neutral/.wav
└── happy/.wav
└── angry/*.wav

Launch the notebook
jupyter notebook voice-emotion-detection.ipynb
Run cells in order
Section 1–2: Load data & inspect label distribution
Section 3: Visualize waveforms & spectrograms
Section 4: Extract audio features
Section 5: Train ML models & compare accuracy
Section 6: View confusion matrix & classification report
Section 7: Launch interactive chat prototype
Save a trained model (example for Random Forest)
import joblib
joblib.dump(rf, "random_forest_emotion_model.pkl")
