import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# ── PATHS ──────────────────────────────────
DATA_DIR = "alphabet_data_v1"
MODEL_PATH = "isl_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# ── LOAD DATA ──────────────────────────────
dfs = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

X = data.drop("label", axis=1).values
y = data["label"].values

# ── LOAD ENCODER & SCALER ──────────────────
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

y_encoded = le.transform(y)
X_scaled = scaler.transform(X)

# ── LOAD MODEL ─────────────────────────────
model = load_model(MODEL_PATH)

# ── PREDICTIONS ────────────────────────────
y_pred_probs = model.predict(X_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── METRICS ────────────────────────────────
accuracy = accuracy_score(y_encoded, y_pred)
precision = precision_score(y_encoded, y_pred, average='weighted')
recall = recall_score(y_encoded, y_pred, average='weighted')
f1 = f1_score(y_encoded, y_pred, average='weighted')

print("\n=== EVALUATION RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ── CLASSIFICATION REPORT ──────────────────
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

# ── CONFUSION MATRIX ───────────────────────
cm = confusion_matrix(y_encoded, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Alphabets Model")

# SAVE IMAGE
plt.savefig("confusion_matrix.png")
plt.show()

print("\n Confusion matrix saved as confusion_matrix.png")