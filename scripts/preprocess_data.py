import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

# Path to ASL data
data_path = r"C:\Users\shrav\Desktop\ASL_Detection_Project\data\asl_alphabet_train"

# Resize target
IMG_SIZE = 64

# Prepare arrays
X = []
y = []

# Get class names
classes = sorted(os.listdir(data_path))
print(f"Classes: {classes}")

# Map class name to index
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Loop through each class folder
print("Loading images...")

for cls in classes:
    folder_path = os.path.join(data_path, cls)
    for img_name in os.listdir(folder_path)[:1000]:  # Limit to 1000 images/class (optional)
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(class_to_idx[cls])

print("Finished loading.")

# Convert to NumPy arrays
X = np.array(X, dtype="float32") / 255.0   # normalize
y = to_categorical(np.array(y), num_classes=len(classes))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save data
os.makedirs("processed_data", exist_ok=True)

with open("processed_data/data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test, classes), f)

print("✅ Data preprocessing complete!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
