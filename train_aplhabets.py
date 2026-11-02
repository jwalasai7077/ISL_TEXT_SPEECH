import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ── SETTINGS ───────────────────────────────
DATASET_DIR = "alphabet_data_v1"   # 🔥 change version if needed
MODEL_NAME = "isl_model.h5"

# ── LOAD DATA ──────────────────────────────
dataframes = []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATASET_DIR, file))
        dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("Total samples:", len(data))

# ── SPLIT FEATURES & LABELS ────────────────
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ── ENCODE LABELS ──────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# ── SCALE FEATURES ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── TRAIN TEST SPLIT ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# ── MODEL ──────────────────────────────────
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(126,)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(len(le.classes_), activation='softmax'))

# ── COMPILE ────────────────────────────────
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── SAVE BEST MODEL ────────────────────────
checkpoint = ModelCheckpoint(
    MODEL_NAME,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ── TRAIN (FIXED 100 EPOCHS) ───────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)

# ── FINAL EVALUATION ───────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

# ── SAVE SCALER & LABEL ENCODER ────────────
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n Training completed for full 100 epochs")
print(" Model + Scaler + Encoder saved")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train", "Val"])
plt.show()

# =========================================================
# LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history['loss'],
    label='Train'
)

plt.plot(
    history.history['val_loss'],
    label='Val'
)

plt.title('Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.savefig("training_loss.png")

plt.show()