import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import random

with open("processed_data/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, classes = pickle.load(f)

model = load_model("models/best_model.h5")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true_labels, y_pred_labels, target_names=classes))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Show random predictions
print("\n🔍 Sample Predictions:")
for i in random.sample(range(len(X_test)), 5):
    plt.imshow(X_test[i])
    plt.title(f"True: {classes[y_true_labels[i]]} | Predicted: {classes[y_pred_labels[i]]}")
    plt.axis('off')
    plt.show()
print("\n🔎 Unique predicted classes by the model:")
unique_preds = np.unique(y_pred_labels)
for idx in unique_preds:
    print(f"→ {idx}: {classes[idx]}")
    
print(f"\nTotal unique classes predicted: {len(unique_preds)}")
