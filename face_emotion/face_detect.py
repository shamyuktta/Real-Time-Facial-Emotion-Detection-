import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import os
import csv
import time
from datetime import datetime
from pathlib import Path
import requests, json
import argparse

# -----------------------------
# Power BI Streaming Dataset Push
# -----------------------------
PUSH_URL = "https://api.powerbi.com/beta/43d1c0f4-7a18-4aeb-bbd4-d5eb093ad1fc/datasets/cde82d09-0143-48cf-9f50-a136e67f37ca/rows?experience=power-bi&key=pS%2F0EdG25imt7Mlgp8wj%2BDBleYDooiybokXxY1%2Fg7Gfj9aJa%2FaaU8Itd26cQt06DbotXKVxBmQwRn3qGbVlkKQ%3D%3D"
BASE_DIR = Path(__file__).resolve().parent.parent
last_push_time = 0
PUSH_INTERVAL_SEC = 3
last_emotion = None

def push_row(emotion, confidence):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "emotion": emotion,
        "confidence": float(confidence),
        "count": 1
    }
    try:
        response = requests.post(PUSH_URL,
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"rows": [row]}))
        print("Power BI response:", response.status_code, response.text)
    except Exception as e:
        print("Push failed:", e)

def safe_push(emotion, confidence):
    global last_push_time, last_emotion
    now = time.time()
    if (emotion != last_emotion) or (now - last_push_time >= PUSH_INTERVAL_SEC):
        push_row(emotion, confidence)
        last_push_time = now
        last_emotion = emotion

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.6,
                    help="Confidence threshold (0.0–1.0), default=0.6")
parser.add_argument("--cam", type=int, default=0, help="Webcam index (default=0)")
args = parser.parse_args()

# -----------------------------
# Load v2 model (RGB 224x224)
# -----------------------------
print("Loading v2 model (RGB 224x224)...")
model = tf.keras.models.load_model(os.path.join(BASE_DIR,'models/face_emotion_model_transfer.keras' ))

labels_path = '../models/labels.json'
if os.path.exists(labels_path):
    with open(labels_path, 'r') as f:
        EMOTIONS = json.load(f)
else:
    num_classes = model.output_shape[-1]
    EMOTIONS = [f"class_{i}" for i in range(num_classes)]
    print(f"labels.json not found. Using fallback labels: {EMOTIONS}")

IMG_SIZE = (224, 224)
print("Model input shape:", model.input_shape)

# -----------------------------
# Haar cascade for face detection
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise RuntimeError("Haar cascade failed to load. Check OpenCV installation.")

# Dictionary to store smoothing buffers per face
face_buffers = {}

# -----------------------------
# Create CSV log file
# -----------------------------
log_path = '../logs/emotion_log.csv'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp', 'emotion', 'confidence'])

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened. Check your camera index or permissions.")

print("Starting webcam... Press 'q' to quit.")

# -----------------------------
# FPS tracking
# -----------------------------
fps_start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    frame_emotions = []  # collect emotions for group mood

    for (x, y, w, h) in faces:
        roi_bgr = frame[y:y+h, x:x+w]
        if roi_bgr.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi_rgb, IMG_SIZE).astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]

        # Use face position as a key (rounded to reduce jitter)
        face_key = (x//10, y//10, w//10, h//10)
        if face_key not in face_buffers:
            face_buffers[face_key] = deque(maxlen=10)

        face_buffers[face_key].append(preds)
        avg_preds = np.mean(face_buffers[face_key], axis=0)

        emotion_idx = int(np.argmax(avg_preds))
        emotion = EMOTIONS[emotion_idx]
        conf = float(np.max(avg_preds))
        conf_pct = conf * 100.0

        if conf >= args.threshold:
            frame_emotions.append(emotion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({conf_pct:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ✅ Log to CSV
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        emotion, f"{conf_pct:.2f}"])

            # ✅ Push to Power BI
            safe_push(emotion, conf_pct)

    # -----------------------------
    # Group mood overlay
    # -----------------------------
    if frame_emotions:
        counts = Counter(frame_emotions)
        dominant = counts.most_common(1)[0][0]
        cv2.putText(frame, f"Group Mood: {dominant}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # -----------------------------
    # FPS calculation
    # -----------------------------
    frame_count += 1
    if frame_count >= 10:
        fps_end_time = time.time()
        fps = frame_count / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Face Emotion Detection (Multi-Face + Group Mood)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended. Predictions logged to:", log_path)

# -----------------------------
# Session summary
# -----------------------------
def summarize_session(log_path):
    emotions = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotions.append(row['emotion'])
    if emotions:
        counts = Counter(emotions)
        dominant = counts.most_common(1)[0]
        print(f"Dominant emotion: {dominant[0]} ({dominant[1]} occurrences)")
    else:
        print("No emotions logged in this session.")

summarize_session(log_path)
