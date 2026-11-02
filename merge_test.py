import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time
import pygame
import tempfile
import os

from gtts import gTTS

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "isl_merge_model.h5"
SCALER_PATH = "merge_scaler.pkl"
LABEL_ENCODER_PATH = "merge_label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

print("\n===================================")
print("MODEL LOADED SUCCESSFULLY")
print("Total Classes:", len(le.classes_))
print("Classes:")
print(le.classes_)
print("===================================\n")

# ==========================================
# TEXT TO SPEECH
# ==========================================

pygame.mixer.init()

def speak(text):
    try:
        if text.strip() == "":
            return

        tts = gTTS(text=text, lang="en")

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        tts.save(temp_file.name)

        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pass

        pygame.mixer.music.unload()

        try:
            os.remove(temp_file.name)
        except:
            pass

    except Exception as e:
        print("TTS Error:", e)

# ==========================================
# MEDIAPIPE HANDS
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

sentence = []
current_prediction = ""
prediction_time = 0
confidence = 0

print("\nControls")
print("SPACE      -> Add word")
print("V          -> Speak sentence")
print("BACKSPACE  -> Delete last word")
print("C          -> Clear sentence")
print("Q          -> Quit\n")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    start_frame = time.time()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    prediction = ""
    confidence = 0

    if results.multi_hand_landmarks:

        landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

        # ==================================
        # FIXED FEATURE LENGTH = 126
        # ==================================

        while len(landmarks) < 126:
            landmarks.append(0)

        landmarks = landmarks[:126]

        feature_vector = np.array(
            landmarks,
            dtype=np.float32
        ).reshape(1, -1)

        # Scale features
        feature_vector = scaler.transform(
            feature_vector
        )

        # LSTM input
        feature_vector = feature_vector.reshape(
            1,
            1,
            126
        )

        start_pred = time.time()

        pred = model.predict(
            feature_vector,
            verbose=0
        )

        prediction_time = (
            time.time() - start_pred
        ) * 1000

        class_idx = np.argmax(pred)

        confidence = (
            pred[0][class_idx] * 100
        )

        prediction = le.inverse_transform(
            [class_idx]
        )[0]

        current_prediction = prediction

    fps = 1 / (
        time.time() - start_frame + 1e-6
    )

    # ==================================
    # DISPLAY
    # ==================================

    cv2.rectangle(
        frame,
        (5, 5),
        (900, 220),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Prediction Time: {prediction_time:.2f} ms",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    sentence_text = " ".join(sentence)

    cv2.putText(
        frame,
        f"Sentence: {sentence_text}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

for h in results.multi_handedness:
    print(h.classification[0].label)
    # ==================================
    # INSTRUCTIONS
    # ==================================

    cv2.putText(
        frame,
        "SPACE:Add  V:Speak  BACKSPACE:Delete  C:Clear  Q:Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Hybrid DNN-LSTM ISL to Speech",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ==================================
    # SPACE -> ADD WORD
    # ==================================

    if key == 32:

        if current_prediction != "":

            if len(sentence) == 0:
                sentence.append(current_prediction)

            elif sentence[-1] != current_prediction:
                sentence.append(current_prediction)

    # ==================================
    # V -> SPEAK
    # ==================================

    elif key == ord('v'):

        text = " ".join(sentence)

        if text.strip():
            speak(text)

    # ==================================
    # DELETE LAST WORD
    # ==================================

    elif key == 8:

        if len(sentence) > 0:
            sentence.pop()

    # ==================================
    # CLEAR SENTENCE
    # ==================================

    elif key == ord('c'):

        sentence.clear()

    # ==================================
    # QUIT
    # ==================================

    elif key == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()