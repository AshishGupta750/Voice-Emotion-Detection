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

## Project Overview

This voice emotion detection project aims to classify spoken utterances into seven emotional categories using audio processing and a deep-learning model. By leveraging an established emotional speech dataset and an LSTM-based neural network, the system demonstrates robust performance on real speech samples.

---

## 1. Dataset Description

* **Source:** The Toronto Emotional Speech Set (TESS) from Kaggle [Download Link](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
* **Speakers:** Two female actors (ages 26 and 64)
* **Utterances:** 200 distinct target words embedded in the carrier phrase “Say the word ‘\_\_\_’”
* **Emotional States:** anger, disgust, fear, happiness, pleasant surprise, sadness, neutral
* **Total Samples:** 2,800 WAV files (200 words × 7 emotions × 2 speakers)

**Folder Hierarchy:**

```
dataset/
├── Actress26/
│   ├── anger/
│   │   ├── word01.wav
│   │   ├── word02.wav
│   │   └── ...
│   ├── happiness/
│   └── ...
└── Actress64/
    ├── surprise/
    ├── neutral/
    └── ...
```

Each emotion folder contains 200 WAV recordings of that emotion spoken by the respective actress.

---

## 2. Output Classes

The model predicts one of the following seven labels:

1. **Anger**
2. **Disgust**
3. **Fear**
4. **Happiness**
5. **Pleasant Surprise**
6. **Sadness**
7. **Neutral**

These classes correspond directly to the actors’ portrayed emotions in the recordings.

---

## 3. Additional Datasets

To expand training data or compare performance, you can explore:

* **EMO-DB** (Berlin database of emotional speech)
* **Ravdess** (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **IEMOCAP** (Interactive Emotional Dyadic Motion Capture)

Reference Kaggle collection: [Speech Emotion Recognition Datasets](https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en)

---

## 4. Dependencies & Libraries

* **Python 3.8+**
* **Pandas**, **NumPy** for data handling
* **Librosa** for audio feature extraction
* **TensorFlow** & **Keras** for model construction
* **Matplotlib** & **Seaborn** for visualizations
* **Scikit-learn** for preprocessing and metrics

Example `requirements.txt`:

```text
pandas
numpy
librosa
matplotlib
seaborn
tensorflow
keras
scikit-learn
```

---

## 5. Neural Network Architecture

We employ a sequential LSTM-based model to capture temporal dependencies in audio features (MFCCs):

1. **Input Layer:** Sequences of 40 MFCC coefficients over time frames
2. **LSTM Blocks:** Two stacked LSTM layers (128 units each, `return_sequences=True` on first layer)
3. **Dropout:** 0.3 between LSTM layers to mitigate overfitting
4. **Dense Layer:** 64 units with ReLU activation
5. **Output Layer:** Softmax activation with 7 units (one per emotion)

**Training Configuration:**

* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam (learning rate = 0.001)
* **Batch Size:** 32
* **Epochs:** 50

---

## 6. Performance Metrics

| Metric           | Result |
| ---------------- | -----: |
| Overall Accuracy |  67.0% |
| Precision (avg.) |   0.68 |
| Recall (avg.)    |   0.67 |
| F1-score (avg.)  |   0.67 |

**Confusion Matrix:** Visual inspection reveals balanced performance across most classes, with slightly lower recall for ‘disgust’ and ‘fear.’

---

## 7. Future Enhancements

* **Data Augmentation:** Time-stretching, pitch-shifting to increase dataset diversity
* **Hybrid Models:** Combine CNNs (on spectrogram images) with LSTMs for richer feature learning
* **Real-Time Processing:** Build a live audio capture interface for on-the-fly emotion detection
* **Hyperparameter Tuning:** Automated search (e.g. Bayesian optimization) for optimal model settings

---

## 8. Usage Instructions

1. **Clone Repo & Install:**

   ```bash
   ```

git clone [https://github.com/](https://github.com/)<USERNAME>/voice-emotion-detection.git
cd voice-emotion-detection
pip install -r requirements.txt

````

2. **Preprocess & Extract Features:**
   ```bash
python extract_features.py --input_dir dataset/ --output_csv features.csv
````

3. **Train Model:**

   ```bash
   ```

python train\_lstm.py --features features.csv --model\_output lstm\_emotion.h5

````

4. **Evaluate:**
   ```bash
python evaluate.py --model lstm_emotion.h5 --test_data features.csv
````

---

Download link: [https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess More Datasets: https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en](https://www.kaggle.com/code/ashishkumar7507754/voice-emotion-detection/edit)

*Prepared by Ashish – July 2025*

