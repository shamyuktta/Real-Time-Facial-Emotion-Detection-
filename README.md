# 🎭 Real-Time Facial Emotion Detection using Deep Learning  

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/TensorFlow-CNN-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/OpenCV-Real--Time-green?style=for-the-badge&logo=opencv">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Emotion%20Classes-10-purple?style=for-the-badge">
</p>

---

## 🌟 Project Overview

This project is a **Real-Time Facial Emotion Detection System** that uses **Deep Learning (CNN)** to recognize human emotions from a **live webcam feed**.

It detects faces, processes facial expressions, and predicts emotions **accurately and stably** using TensorFlow and OpenCV.  
Built with strong fundamentals — no shortcuts, no gimmicks.

> 🎯 Goal: **High accuracy + stable emotion prediction in real time**

---

## 😄 Emotions Detected

The system currently supports **10 emotion classes**:

- 😡 Angry  
- 😕 Confused  
- 😒 Contempt  
- 🤢 Disgust  
- 🤩 Excited  
- 😨 Fear  
- 😀 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

---

## 🧠 How It Works (Pipeline)

### 🎥 1. Webcam Capture
- Captures live video frames using OpenCV.

### 👤 2. Face Detection
- Detects faces using **Haar Cascade Classifier**.
- Crops each detected face accurately.

### 🖼️ 3. Preprocessing
Each face is:
- Converted to **grayscale**
- Resized to **48×48**
- Normalized (pixel values between 0 and 1)
- Reshaped for CNN input

### 🤖 4. Emotion Prediction
- Processed face is passed to a **trained CNN model**
- Model outputs probability scores for each emotion

### 🔄 5. Emotion Smoothing (Stability Boost)
- Predictions are stored in a buffer (`deque`)
- Final emotion is selected using **majority voting**
- Prevents flickering and unstable emotion labels

### 🖥️ 6. Real-Time Display
- Draws face bounding box
- Displays emotion label + confidence percentage

---

## ✨ Key Features

- 🎥 Real-time emotion detection  
- 🧠 CNN-based deep learning model (TensorFlow/Keras)  
- 📊 Confidence score display  
- 🔄 Emotion smoothing for stable output  
- ⚡ Fast and lightweight execution  
- 🧩 Clean and modular project structure  
- 💻 Works smoothly on Python 3.10  

---

## 🧰 Tech Stack

| Category | Tools |
|-------|------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Face Detection | Haar Cascade |
| Utilities | NumPy, Deque |

---

## 📁 Project Structure
team_mood_project/
│
├── face_emotion/
│ └── face_detect.py # Real-time detection code
│
├── models/
│ └── emotion_model.h5 # Trained CNN model
│
├── logs/ # Runtime logs
│
├── check_env.py # Environment check script
├── requirements.txt # Dependencies
└── README.md

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your system.

---

### 🧩 Prerequisites

Make sure the following are installed:

- 🐍 **Python 3.10.x** (recommended: 3.10.19)
- 📷 **Webcam** (for real-time emotion detection)
- 💻 Windows / Linux / macOS

> ⚠️ TensorFlow is most stable with Python 3.10. Avoid newer Python versions.

---

### 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd team_mood_project

🧪 Step 2: Create a Virtual Environment (Highly Recommended)
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate


You should see (venv) in the terminal — that means you’re inside the environment ✅

📦 Step 3: Install Required Dependencies
pip install --upgrade pip
pip install tensorflow opencv-python numpy


Or install everything at once (if requirements.txt is available):

pip install -r requirements.txt

🔍 Step 4: Verify Installations (Optional but Recommended)
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import numpy as np; print(np.__version__)"


If versions print without errors → setup is successful 🎉

▶️ Step 6: Run the Application
python face_emotion/face_detect.py

✅ Expected Output

📷 Webcam opens

👤 Face detected in real time

😄 Emotion label with confidence percentage displayed

⛔ Press ESC to exit

🧠 Model Training (Deep Learning Pipeline)

This section explains how the emotion recognition model was trained.

📊 Datasets Used

The CNN model was trained using a combined facial emotion dataset:

FER-2013

CK+

Indian Face Dataset

All datasets were merged to improve:

Emotion diversity

Ethnic representation

Real-world accuracy

🖼️ Data Preprocessing

Each facial image was processed as follows:

Converted to grayscale

Resized to 48 × 48

Pixel values normalized to range [0, 1]

Reshaped to (48, 48, 1) for CNN input

Labeled according to emotion class

This ensures uniform input across all datasets.


