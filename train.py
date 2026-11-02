import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LSTM,
)

from tensorflow.keras.models import Sequential


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "gesture_data"

MODEL_NAME = "gesture_lstm_dnn_model.h5"

SCALER_NAME = "word_scaler.pkl"
LABEL_ENCODER_NAME = "word_label_encoder.pkl"

CONFIG_NAME = "word_model_config.json"

SEQUENCE_LENGTH = 20
WINDOW_STRIDE = 2

TEST_SIZE = 0.2
VAL_SIZE = 0.1

EPOCHS = 100
BATCH_SIZE = 16

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# CREATE SEQUENCE WINDOWS
# =========================================================

def build_windows(features, label, sequence_length, stride):

    sequences = []
    labels = []

    for start in range(
        0,
        len(features) - sequence_length + 1,
        stride,
    ):

        end = start + sequence_length

        sequences.append(
            features[start:end]
        )

        labels.append(label)

    return sequences, labels


# =========================================================
# LOAD DATASET
# =========================================================

def load_sequence_dataset(
    dataset_dir,
    sequence_length,
    stride,
):

    all_sequences = []
    all_labels = []

    print("\nLoading gesture sequences...\n")

    for filename in sorted(os.listdir(dataset_dir)):

        if not filename.endswith(".csv"):
            continue

        path = os.path.join(
            dataset_dir,
            filename,
        )

        df = pd.read_csv(path).dropna()

        if "label" not in df.columns:

            print(
                f"Skipping {filename}: "
                f"missing label column"
            )

            continue

        if len(df) < sequence_length:

            print(
                f"Skipping {filename}: "
                f"only {len(df)} frames, "
                f"need at least {sequence_length}"
            )

            continue

        label_values = (
            df["label"]
            .astype(str)
            .unique()
        )

        label = label_values[0]

        features = (
            df.drop(columns=["label"])
            .values
            .astype("float32")
        )

        sequences, labels = build_windows(
            features,
            label,
            sequence_length,
            stride,
        )

        print(
            f"{filename:20s} -> "
            f"{len(features):3d} frames, "
            f"{len(sequences):3d} windows"
        )

        all_sequences.extend(sequences)
        all_labels.extend(labels)

    if not all_sequences:

        raise ValueError(
            "No valid gesture sequences found in dataset."
        )

    X = np.array(
        all_sequences,
        dtype="float32",
    )

    y = np.array(all_labels)

    return X, y


# =========================================================
# FEATURE SCALING
# =========================================================

def scale_sequences(
    train_x,
    val_x,
    test_x,
):

    feature_count = train_x.shape[-1]

    scaler = StandardScaler()

    train_2d = train_x.reshape(
        -1,
        feature_count,
    )

    val_2d = val_x.reshape(
        -1,
        feature_count,
    )

    test_2d = test_x.reshape(
        -1,
        feature_count,
    )

    scaler.fit(train_2d)

    train_scaled = (
        scaler.transform(train_2d)
        .reshape(train_x.shape)
    )

    val_scaled = (
        scaler.transform(val_2d)
        .reshape(val_x.shape)
    )

    test_scaled = (
        scaler.transform(test_2d)
        .reshape(test_x.shape)
    )

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler,
    )


# =========================================================
# BUILD HYBRID LSTM-DNN MODEL
# =========================================================

def build_model(
    sequence_length,
    feature_count,
    num_classes,
):

    model = Sequential([

        # =================================================
        # INPUT LAYER
        # =================================================

        Input(
            shape=(
                sequence_length,
                feature_count,
            )
        ),

        # =================================================
        # LSTM FEATURE EXTRACTION
        # =================================================

        LSTM(
            128,
            return_sequences=True,
        ),

        Dropout(0.3),

        LSTM(64),

        Dropout(0.3),

        # =================================================
        # DNN CLASSIFICATION
        # =================================================

        Dense(
            256,
            activation="relu",
        ),

        BatchNormalization(),

        Dropout(0.4),

        Dense(
            128,
            activation="relu",
        ),

        BatchNormalization(),

        Dropout(0.3),

        Dense(
            64,
            activation="relu",
        ),

        # =================================================
        # OUTPUT LAYER
        # =================================================

        Dense(
            num_classes,
            activation="softmax",
        ),

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],

    )

    return model


# =========================================================
# LOAD DATA
# =========================================================

X, y = load_sequence_dataset(

    DATASET_DIR,

    SEQUENCE_LENGTH,

    WINDOW_STRIDE,

)

print(f"\nDataset Shape: {X.shape}")


# =========================================================
# LABEL ENCODING
# =========================================================

le = LabelEncoder()

y_enc = le.fit_transform(y)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_enc,

    test_size=TEST_SIZE,

    random_state=SEED,

    stratify=y_enc,

)

val_ratio_within_train = (
    VAL_SIZE / (1.0 - TEST_SIZE)
)

X_train, X_val, y_train, y_val = train_test_split(

    X_train,
    y_train,

    test_size=val_ratio_within_train,

    random_state=SEED,

    stratify=y_train,

)


# =========================================================
# NORMALIZATION
# =========================================================

X_train, X_val, X_test, scaler = scale_sequences(

    X_train,
    X_val,
    X_test,

)

print("\nTrain Shape:", X_train.shape)

print("Validation Shape:", X_val.shape)

print("Test Shape:", X_test.shape)


# =========================================================
# CLASS WEIGHTS
# =========================================================

classes = np.unique(y_train)

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=y_train,

)

class_weights = {

    int(class_id): float(weight)

    for class_id, weight in zip(
        classes,
        weights,
    )

}


# =========================================================
# BUILD MODEL
# =========================================================

model = build_model(

    sequence_length=SEQUENCE_LENGTH,

    feature_count=X_train.shape[-1],

    num_classes=len(le.classes_),

)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

lr_reduce = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=1e-5,

    verbose=1,

)

checkpoint = ModelCheckpoint(

    MODEL_NAME,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1,

)


# =========================================================
# TRAIN MODEL
# =========================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val,
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[

        lr_reduce,

        checkpoint,

    ],

    verbose=1,

)


# =========================================================
# TRAINING ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy",

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy",

)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_accuracy.png"
)

plt.show()


# =========================================================
# TRAINING LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(

    history.history["loss"],

    label="Training Loss",

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss",

)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_loss.png"
)

plt.show()


# =========================================================
# LOAD BEST MODEL
# =========================================================

best_model = tf.keras.models.load_model(
    MODEL_NAME
)


# =========================================================
# PREDICTIONS
# =========================================================

y_pred = best_model.predict(

    X_test,

    verbose=0,

)

y_pred_labels = np.argmax(

    y_pred,

    axis=1,

)


# =========================================================
# EVALUATION
# =========================================================

test_accuracy = accuracy_score(

    y_test,
    y_pred_labels,
)

print(
    "\nTest Accuracy:",
    test_accuracy,
)

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,

        y_pred_labels,

        labels=np.arange(
            len(le.classes_)
        ),

        target_names=le.classes_,

        zero_division=0,

    )

)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(

    y_test,
    y_pred_labels,

)

plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=le.classes_,

    yticklabels=le.classes_,

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# =========================================================
# SAVE FILES
# =========================================================

with open(
    SCALER_NAME,
    "wb",
) as f:

    pickle.dump(
        scaler,
        f,
    )

with open(
    LABEL_ENCODER_NAME,
    "wb",
) as f:

    pickle.dump(
        le,
        f,
    )

with open(
    CONFIG_NAME,
    "w",
    encoding="utf-8",
) as f:

    json.dump(

        {

            "dataset_dir": DATASET_DIR,

            "sequence_length": SEQUENCE_LENGTH,

            "window_stride": WINDOW_STRIDE,

            "feature_count": int(
                X_train.shape[-1]
            ),

            "model_name": MODEL_NAME,

            "scaler_name": SCALER_NAME,

            "label_encoder_name":
                LABEL_ENCODER_NAME,

            "class_names":
                le.classes_.tolist(),

            "epochs_ran":
                len(history.history["loss"]),

            "seed": SEED,

            "test_accuracy":
                float(test_accuracy),

        },

        f,

        indent=2,

    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\nSaved Files:")

print(f"- {MODEL_NAME}")

print(f"- {SCALER_NAME}")

print(f"- {LABEL_ENCODER_NAME}")

print(f"- {CONFIG_NAME}")

print("- training_accuracy.png")

print("- training_loss.png")

print("- confusion_matrix.png")

print("\nTraining Complete Successfully.")