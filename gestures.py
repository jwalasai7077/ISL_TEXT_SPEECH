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

model = tf.keras.models.load_model("isl_merge_model.h5")

with open("merge_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("merge_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Model Loaded")
print("Classes:", len(le.classes_))

# ==========================================
# SPEECH
# ==========================================

pygame.mixer.init()

def speak(text):

    if text.strip() == "":
        return

    try:
        tts = gTTS(text=text, lang="en")

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        tts.save(temp.name)

        pygame.mixer.music.load(temp.name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pass

        pygame.mixer.music.unload()

        os.remove(temp.name)

    except Exception as e:
        print("Speech Error:", e)

# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ==========================================
# EXACT SAME FEATURE EXTRACTION
# AS DATASET CREATION
# ==========================================

def extract_features_two_hands(
        hand_landmarks_list,
        handedness_list):

    left = [0.0] * 63
    right = [0.0] * 63

    for i, hand_landmarks in enumerate(
            hand_landmarks_list):

        coords = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        coords -= coords[0]

        scale = (
            np.max(np.abs(coords))
            + 1e-6
        )

        coords /= scale

        features = coords.flatten().tolist()

        hand_label = (
            handedness_list[i]
            .classification[0]
            .label
        )

        if hand_label == "Left":
            left = features
        else:
            right = features

    return left + right

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

sentence = []
current_word = ""

print("\nControls")
print("SPACE = Add Word")
print("V = Speak")
print("BACKSPACE = Delete Last Word")
print("C = Clear Sentence")
print("Q = Quit\n")

# ==========================================
# LOOP
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
    pred_time = 0

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        feature_vector = extract_features_two_hands(
            results.multi_hand_landmarks,
            results.multi_handedness
        )

        feature_vector = np.array(
            feature_vector,
            dtype=np.float32
        ).reshape(1, -1)

        feature_vector = scaler.transform(
            feature_vector
        )

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

        pred_time = (
            time.time() - start_pred
        ) * 1000

        idx = np.argmax(pred)

        prediction = le.inverse_transform(
            [idx]
        )[0]

        confidence = (
            pred[0][idx] * 100
        )

        current_word = prediction

    fps = 1 / (
        time.time() - start_frame + 1e-6
    )

    sentence_text = " ".join(sentence)

    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,0,0),
        2
    )

    cv2.putText(
        frame,
        f"Prediction Time: {pred_time:.2f} ms",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Sentence: {sentence_text}",
        (20,210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2
    )

    cv2.putText(
        frame,
        "SPACE:Add  V:Speak  BACKSPACE:Delete  C:Clear  Q:Quit",
        (10, frame.shape[0]-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0,255,0),
        2
    )

    cv2.imshow(
        "ISL Recognition System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        if current_word != "":

            if len(sentence) == 0:
                sentence.append(current_word)

            elif sentence[-1] != current_word:
                sentence.append(current_word)

    # V
    elif key == ord('v'):

        text = " ".join(sentence)

        if text:
            speak(text)

    # BACKSPACE
    elif key == 8:

        if len(sentence) > 0:
            sentence.pop()

    # CLEAR
    elif key == ord('c'):

        sentence.clear()

    # QUIT
    elif key == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()