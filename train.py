"""Train the SignSense MLP on collected landmarks.

Usage:
    python train.py
Outputs:
    models/signsense_mlp.keras, models/labels.json
    results/training_curves.png, results/confusion_matrix.png, results/report.txt
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from utils import normalize

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---- Load and prepare data ----
df = pd.read_csv("data/landmarks.csv")
print("Samples per letter:\n", df["label"].value_counts().sort_index().to_string())

X = np.array([normalize(row) for row in df.iloc[:, 2:].values], dtype="float32")
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])
n_classes = len(encoder.classes_)

# 80% train / 10% val / 10% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# ---- Model ----
model = keras.Sequential([
    keras.layers.Input(shape=(63,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(n_classes, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True)],
    verbose=2,
)

# ---- Evaluate ----
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
report = classification_report(y_test, y_pred, target_names=[c.upper() for c in encoder.classes_])
print(report)
with open("results/report.txt", "w") as f:
    f.write(f"Test accuracy: {test_acc:.4f}\n\n{report}")

# Training curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=150); plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay(cm, display_labels=[c.upper() for c in encoder.classes_]).plot(
    ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

# ---- Save ----
model.save("models/signsense_mlp.keras")
with open("models/labels.json", "w") as f:
    json.dump(list(encoder.classes_), f)
print("Saved model, labels and result plots.")
