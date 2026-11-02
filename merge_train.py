import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# LOAD ALL CSV FILES
# ======================================

DATASET_PATH = "merged_dataset"

X = []
y = []

for file in os.listdir(DATASET_PATH):

    if file.endswith(".csv"):

        print("Loading:", file)

        df = pd.read_csv(
            os.path.join(DATASET_PATH, file)
        )

        # Features
        features = df.drop(columns=["label"])

        # Labels
        labels = df["label"]

        X.extend(features.values)
        y.extend(labels.values)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nSamples:", len(X))
print("Classes:", len(np.unique(y)))
print("Class Names:")
print(sorted(np.unique(y)))

# ======================================
# LABEL ENCODING
# ======================================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open("merge_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ======================================
# FEATURE SCALING
# ======================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

with open("merge_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ======================================
# RESHAPE FOR LSTM
# ======================================

X = X.reshape(X.shape[0], 1, X.shape[1])

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# ======================================
# DNN + LSTM MODEL
# ======================================

model = Sequential()

model.add(LSTM(
    128,
    return_sequences=True,
    input_shape=(X.shape[1], X.shape[2])
))

model.add(Dropout(0.3))

model.add(LSTM(64))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(
    len(le.classes_),
    activation='softmax'
))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================================
# TRAIN
# ======================================

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32
)

# ======================================
# SAVE MODEL
# ======================================

model.save("isl_merge_model.h5")

# ======================================
# ACCURACY PLOT
# ======================================

plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.savefig("training_accuracy.png")
plt.show()

# ======================================
# LOSS PLOT
# ======================================

plt.figure(figsize=(8,6))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.savefig("training_loss.png")
plt.show()

# ======================================
# PREDICTIONS
# ======================================

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

# ======================================
# CONFUSION MATRIX
# ======================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20,20))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

# ======================================
# CLASSIFICATION REPORT
# ======================================

report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
)

print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# ======================================
# DATASET DISTRIBUTION
# ======================================

class_counts = pd.Series(y).value_counts()

plt.figure(figsize=(15,6))

class_counts.plot(kind='bar')

plt.title("Dataset Distribution")
plt.xlabel("Classes")
plt.ylabel("Samples")

plt.tight_layout()

plt.savefig("dataset_distribution.png")
plt.show()

print("Training Complete")