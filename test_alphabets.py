import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
import time
import pyttsx3   # kept as-is (not used now, but not removed as per your request)

# 🔊 NEW IMPORTS (for updated speak)
from gtts import gTTS
import pygame
import os
import tempfile

# ── LOAD MODEL ─────────────────────────────
model = load_model("isl_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ── TEXT TO SPEECH (UPDATED ONLY THIS PART) ────────
pygame.mixer.init()

def speak(text):
    if text.strip() == "":
        return

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            filename = fp.name

        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        os.remove(filename)

    except Exception as e:
        print("TTS Error:", e)

# ── MEDIAPIPE ──────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ── FEATURE EXTRACTION ─────────────────────
def extract_features(hand_landmarks_list, handedness_list):
    left = [0.0]*63
    right = [0.0]*63

    for i, hand_landmarks in enumerate(hand_landmarks_list):
        coords = np.array([[lm.x, lm.y, lm.z]
                           for lm in hand_landmarks.landmark], dtype=np.float32)

        coords -= coords[0]
        scale = np.max(np.abs(coords)) + 1e-6
        coords /= scale

        features = coords.flatten().tolist()
        label = handedness_list[i].classification[0].label

        if label == "Left":
            left = features
        else:
            right = features

    return left + right

# ── CAMERA ─────────────────────────────────
cap = cv2.VideoCapture(0)

# ── SETTINGS ───────────────────────────────
CONFIDENCE_THRESHOLD = 0.8
LETTER_DELAY = 3.0
WORD_DELAY = 5.0
SENTENCE_DELAY = 10.0

current_word = ""
current_sentence = ""
paragraph = ""

last_letter = ""
last_letter_time = 0
last_detect_time = time.time()

print("\nControls:")
print("W = speak word | S = speak sentence | P = speak paragraph")
print("D = delete letter | C = clear | Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_char = ""
    confidence = 0

    if results.multi_hand_landmarks:
        last_detect_time = time.time()

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

        features = extract_features(
            results.multi_hand_landmarks,
            results.multi_handedness
        )

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        preds = model.predict(features, verbose=0)
        confidence = np.max(preds)

        if confidence > CONFIDENCE_THRESHOLD:
            predicted_char = le.inverse_transform([np.argmax(preds)])[0]

    current_time = time.time()

    # ── LETTER BUILDING ─────────────────────
    if predicted_char != "":
        if (current_time - last_letter_time > LETTER_DELAY) and (predicted_char != last_letter):
            current_word += predicted_char
            last_letter = predicted_char
            last_letter_time = current_time

    # ── WORD COMPLETE ───────────────────────
    if (time.time() - last_detect_time > WORD_DELAY) and current_word != "":
        current_sentence += current_word + " "
        current_word = ""
        last_letter = ""

    # ── SENTENCE COMPLETE ───────────────────
    if (time.time() - last_detect_time > SENTENCE_DELAY) and current_sentence != "":
        current_sentence = current_sentence.strip() + ". "
        paragraph += current_sentence
        current_sentence = ""

    # ── UI ─────────────────────────────────
    cv2.putText(frame, f"Word: {current_word}",
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,255,0), 2)

    cv2.putText(frame, f"Sentence: {current_sentence}",
                (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)

    cv2.putText(frame, f"Paragraph: {paragraph}",
                (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200,200,200), 2)

    cv2.putText(frame, f"Pred: {predicted_char} ({confidence:.2f})",
                (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,0), 2)

    cv2.imshow("ISL Voice Button System", frame)

    key = cv2.waitKey(1) & 0xFF

    # 🔊 SPEAK CONTROLS
    if key == ord('w'):
        speak(current_word)

    elif key == ord('s'):
        speak(current_sentence)

    elif key == ord('p'):
        speak(paragraph)

    # DELETE LETTER
    elif key == ord('d'):
        if len(current_word) > 0:
            current_word = current_word[:-1]

    # CLEAR
    elif key == ord('c'):
        current_word = ""
        current_sentence = ""
        paragraph = ""

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()