from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -----------------------------
# Data generator for RGB images
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255)

gen = datagen.flow_from_directory(
    "datasets/combined",
    target_size=(224,224),          # v2 model expects 224x224
    color_mode="rgb",               # always RGB now
    batch_size=1,
    class_mode="categorical"
)

print("Class indices:", gen.class_indices)

# -----------------------------
# Extra: print counts per folder
# -----------------------------
root = "datasets/combined"
counts = {}
for label in sorted(os.listdir(root)):
    path = os.path.join(root, label)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith((".jpg",".jpeg",".png"))])
        counts[label] = count
        print(f"{label}: {count} images")

# -----------------------------
# Imbalance check
# -----------------------------
if counts:
    max_count = max(counts.values())
    min_count = min(counts.values())
    print("\nDataset imbalance check:")
    print(f"Most samples: {max_count}, Fewest samples: {min_count}")
    for label, count in counts.items():
        if count < 0.5 * max_count:
            print(f"⚠️ {label} has significantly fewer samples ({count}) compared to max ({max_count})")
