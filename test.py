import json
import pickle
import re
import threading
import time
from collections import Counter, deque

import cv2
import language_tool_python
import mediapipe as mp
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model


MODEL_NAME = "gesture_lstm_model.h5"
SCALER_NAME = "word_scaler.pkl"
LABEL_ENCODER_NAME = "word_label_encoder.pkl"
CONFIG_NAME = "word_model_config.json"


with open(CONFIG_NAME, "r", encoding="utf-8") as f:
    model_config = json.load(f)

MODEL_NAME = model_config.get("model_name", MODEL_NAME)
SCALER_NAME = model_config.get("scaler_name", SCALER_NAME)
LABEL_ENCODER_NAME = model_config.get("label_encoder_name", LABEL_ENCODER_NAME)
SEQUENCE_LENGTH = int(model_config["sequence_length"])
FEATURE_COUNT = int(model_config["feature_count"])

model = load_model(MODEL_NAME)

with open(SCALER_NAME, "rb") as f:
    scaler = pickle.load(f)

with open(LABEL_ENCODER_NAME, "rb") as f:
    le = pickle.load(f)


tool = language_tool_python.LanguageTool("en-US")


def correct_grammar(text):
    if not text.strip():
        return ""

    try:
        return tool.correct(text)
    except Exception:
        return text


def improve_structure(text):
    words = text.lower().split()
    result = []
    i = 0

    while i < len(words):
        if i + 1 < len(words):
            if words[i] == "you" and words[i + 1] in ["good", "bad"]:
                result.append("you are")
                result.append(words[i + 1])
                i += 2
                continue

            if words[i] == "i" and words[i + 1] in ["good", "bad"]:
                result.append("i am")
                result.append(words[i + 1])
                i += 2
                continue

        result.append(words[i])
        i += 1

    return " ".join(result)


def final_nlp_pipeline(text):
    if not text.strip():
        return ""

    text = improve_structure(text)
    text = correct_grammar(text)

    parts = re.split(
        r"\b(hello|hi|thank you|i love you|sorry)\b",
        text,
        flags=re.IGNORECASE,
    )

    sentences = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.lower() in ["hello", "hi", "thank you", "i love you", "sorry"]:
            if current:
                sentences.append(current.strip())
                current = ""
            sentences.append(part)
        else:
            current += " " + part

    if current:
        sentences.append(current.strip())

    final = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence = sentence[0].upper() + sentence[1:]
        final.append(sentence)

    return ". ".join(final) + "."


engine = pyttsx3.init()
engine.setProperty("rate", 160)


def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


def extract_features(hand_landmarks_list, handedness_list):
    left = [0.0] * 63
    right = [0.0] * 63

    for i, hand_landmarks in enumerate(hand_landmarks_list):
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )

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


CONF_THRESHOLD = 0.75
BUFFER_RESET_DELAY = 0.8
prediction_buffer = deque(maxlen=10)
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

sentence = []
last_word = ""
last_added_time = 0
ADD_DELAY = 1.5

last_update_time = time.time()
last_hand_time = 0.0
GRAMMAR_DELAY = 2.0

cap = cv2.VideoCapture(0)

print("\n=== Gesture Sentence System (LSTM + NLP) ===")
print("V = speak | C = clear | BACKSPACE = delete | Q = quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "NO HAND"
    confidence = 0.0

    if result.multi_hand_landmarks:
        last_hand_time = now

        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

        features = extract_features(
            result.multi_hand_landmarks,
            result.multi_handedness,
        )

        if len(features) == FEATURE_COUNT:
            sequence_buffer.append(features)

        if len(sequence_buffer) == SEQUENCE_LENGTH:
            window = np.array(sequence_buffer, dtype=np.float32)
            window_scaled = scaler.transform(window).reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)

            preds = model.predict(window_scaled, verbose=0)[0]
            confidence = float(np.max(preds))

            if confidence > CONF_THRESHOLD:
                label = le.inverse_transform([np.argmax(preds)])[0]
                prediction_buffer.append(label)

                stable_word = Counter(prediction_buffer).most_common(1)[0][0]
                prediction = stable_word

                if stable_word != last_word and (now - last_added_time > ADD_DELAY):
                    sentence.append(stable_word)
                    last_word = stable_word
                    last_added_time = now
                    last_update_time = now
            else:
                prediction = "UNCERTAIN"
                prediction_buffer.clear()
        else:
            prediction = f"WARMING {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
    else:
        if now - last_hand_time > BUFFER_RESET_DELAY:
            sequence_buffer.clear()
            prediction_buffer.clear()

    raw_sentence = " ".join(sentence).replace("_", " ")

    if now - last_update_time > GRAMMAR_DELAY:
        corrected_sentence = final_nlp_pipeline(raw_sentence)
    else:
        corrected_sentence = raw_sentence

    cv2.rectangle(frame, (0, 0), (640, 70), (20, 20, 20), -1)

    cv2.putText(
        frame,
        prediction.upper(),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"{confidence:.2f}",
        (520, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 200, 200),
        2,
    )

    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(
        frame,
        corrected_sentence,
        (10, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Gesture Sentence System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("v"):
        if corrected_sentence:
            speak_async(corrected_sentence)

    elif key == ord("c"):
        sentence = []
        last_word = ""
        prediction_buffer.clear()
        sequence_buffer.clear()

    elif key == 8:
        if sentence:
            sentence.pop()

cap.release()
cv2.destroyAllWindows()
hands.close()
