import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

# -----------------------------
# Paths and sanity checks
# -----------------------------
MODEL_PATH = Path("face_emotion_model_transfer.keras")
LABELS_PATH = Path("labels.json")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Missing labels file: {LABELS_PATH.resolve()}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH.resolve()}")

with open(LABELS_PATH, "r") as f:
    emotion_labels = json.load(f)

# -----------------------------
# Load model WITHOUT compiling
# -----------------------------
# This avoids needing the custom loss during deserialization
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully (compile=False).")

# -----------------------------
# Preprocess function
# -----------------------------
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# Predict function
# -----------------------------
def predict_emotion(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    pred_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return emotion_labels[pred_class], confidence

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    test_image = Path("test.jpg")  # replace with your image filename
    if not test_image.exists():
        raise FileNotFoundError(f"Missing test image: {test_image.resolve()}")
    emotion, conf = predict_emotion(test_image)
    print(f"Predicted Emotion: {emotion} (confidence: {conf:.2f})")
