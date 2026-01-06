import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import albumentations as A

# -----------------------------
# Dataset path (local)
# -----------------------------
DATASET_PATH = "datasets/combined"  # ✅ update to your local folder

# -----------------------------
# Load dataset from folders (RGB)
# -----------------------------
print("Loading image dataset...")
X, y = [], []
emotion_labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

for label in emotion_labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224)).astype('float32') / 255.0
        X.append(img)
        y.append(label_map[label])

X, y = np.array(X), np.array(y)
print("Total samples:", X.shape[0])

# -----------------------------
# Train/val split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# -----------------------------
# Augmentation
# -----------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.GaussianBlur(p=0.2),
    A.RandomGamma(p=0.2)
])

X_train_aug, y_train_aug = [], []
low_classes = ['contempt', 'disgust', 'excited', 'confused']

for img, label in zip(X_train, y_train):
    X_train_aug.append(img)
    y_train_aug.append(label)
    if emotion_labels[label] in low_classes:
        for _ in range(3):
            img_aug = augment(image=img)['image']
            X_train_aug.append(img_aug)
            y_train_aug.append(label)

X_train_aug, y_train_aug = np.array(X_train_aug), np.array(y_train_aug)
print("Training samples after augmentation:", X_train_aug.shape[0])

# -----------------------------
# MobileNetV2 Transfer Learning Model
# -----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
preds = Dense(len(emotion_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# -----------------------------
# Compile with standard loss
# -----------------------------
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Class weights
# -----------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    ModelCheckpoint('face_emotion_model_transfer.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# -----------------------------
# Train model
# -----------------------------
print("Training model...")
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# -----------------------------
# Save model + labels
# -----------------------------
model.save('face_emotion_model_transfer.keras')
with open('labels.json', 'w') as f:
    json.dump(emotion_labels, f)
print("Model and labels saved.")

# -----------------------------
# Plot training curves
# -----------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved.")

# -----------------------------
# Confusion Matrix + Report
# -----------------------------
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_val, y_pred_classes, target_names=emotion_labels))

cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved.")
