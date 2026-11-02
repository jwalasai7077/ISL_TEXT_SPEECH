import cv2
import numpy as np
import mediapipe as mp
import pickle
import threading
import tempfile
import os
import pygame
import time

from collections import Counter, deque
from gtts import gTTS
from tensorflow.keras.models import load_model


# ==========================================================
# MERGED MODEL INITIALIZATION
# ==========================================================

print("Loading merged model...")

MODEL_PATH = "isl_merge_model.h5"
SCALER_PATH = "merge_scaler.pkl"
LABEL_ENCODER_PATH = "merge_label_encoder.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

CLASS_NAMES = [str(label) for label in label_encoder.classes_]
ALPHABET_LABELS = {label for label in CLASS_NAMES if len(label) == 1 and label.isalpha()}
GESTURE_LABELS = set(CLASS_NAMES) - ALPHABET_LABELS

if not ALPHABET_LABELS or not GESTURE_LABELS:
    raise SystemExit("Merged model must contain both alphabet and gesture classes.")

LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
ALPHABET_INDICES = [LABEL_TO_INDEX[label] for label in CLASS_NAMES if label in ALPHABET_LABELS]
GESTURE_INDICES = [LABEL_TO_INDEX[label] for label in CLASS_NAMES if label in GESTURE_LABELS]

FEATURE_COUNT = int(getattr(scaler, "n_features_in_", 126))

print("Merged model loaded successfully.")
print(f"Alphabet classes: {len(ALPHABET_LABELS)}")
print(f"Gesture classes: {len(GESTURE_LABELS)}")


# ==========================================================
# TEXT TO SPEECH
# ==========================================================

pygame.mixer.init()


def speak(text):
    if not text.strip():
        return

    filename = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            filename = fp.name

        tts = gTTS(text=text, lang="en")
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)

        pygame.mixer.music.unload()
    except Exception as exc:
        print("Speech Error:", exc)
    finally:
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass


def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


# ==========================================================
# MEDIAPIPE HANDS
# ==========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.70,
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
        coords /= (np.max(np.abs(coords)) + 1e-6)
        features = coords.flatten().tolist()

        hand_label = handedness_list[i].classification[0].label

        if hand_label == "Left":
            left = features
        else:
            right = features

    return left + right


# ==========================================================
# UI HELPERS
# ==========================================================

def format_prediction(label):
    if label in ALPHABET_LABELS:
        return label.upper()
    return label.replace("_", " ")


def compose_sentence_text(sentence_words, active_word=""):
    tokens = [word.replace("_", " ") for word in sentence_words]

    if active_word.strip():
        tokens.append(active_word.strip())

    return " ".join(tokens).strip()


def compose_saved_sentence_text(sentence_words):
    return " ".join(word.replace("_", " ") for word in sentence_words).strip()


def format_sentence_for_paragraph(text):
    text = text.strip()
    if not text:
        return ""

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?":
        text += "."

    return text


def compose_paragraph_text(paragraph_sentences):
    return " ".join(sentence.strip() for sentence in paragraph_sentences if sentence.strip()).strip()


def draw_wrapped_text(
    image,
    text,
    origin,
    max_width,
    line_height,
    font,
    scale,
    color,
    thickness,
):
    words = text.split()
    if not words:
        return

    x, y = origin
    line = ""

    for word in words:
        candidate = word if not line else f"{line} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]

        if width <= max_width:
            line = candidate
            continue

        cv2.putText(image, line, (x, y), font, scale, color, thickness)
        y += line_height
        line = word

    if line:
        cv2.putText(image, line, (x, y), font, scale, color, thickness)


# ==========================================================
# SETTINGS
# ==========================================================

current_mode = "ALPHABET"

CONFIDENCE_THRESHOLD = 0.80
PREDICTION_DELAY = 1.5
BUFFER_RESET_DELAY = 0.8
MIN_STABLE_VOTES = 5

prediction_buffer = deque(maxlen=10)

sentence = []
paragraph = []
current_word = ""
current_prediction = ""

candidate_prediction = ""
candidate_since = 0.0
candidate_committed = False
last_hand_time = 0.0


# ==========================================================
# CAMERA / WINDOW
# ==========================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

WINDOW_NAME = "Merged Dual Mode ISL Translator"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1600, 900)

LEFT_PANEL = 420
WINDOW_W = 1600
WINDOW_H = 900
CAMERA_W = WINDOW_W - LEFT_PANEL
CAMERA_H = WINDOW_H

print("Initialization complete.")
print("A = alphabet mode | G = gesture mode | SPACE = save word | P = save sentence to paragraph | V = speak | D = delete | C = clear | Q = quit")


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    now = time.time()

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAMERA_W, CAMERA_H))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    ui = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    ui[:, :LEFT_PANEL] = (20, 20, 20)
    ui[:, LEFT_PANEL:] = frame

    display_prediction = "NO HAND"
    confidence = 0.0
    hold_status = f"{PREDICTION_DELAY:.1f}s"

    if results.multi_hand_landmarks:
        last_hand_time = now

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                ui[:, LEFT_PANEL:],
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

        features = extract_features(
            results.multi_hand_landmarks,
            results.multi_handedness,
        )

        x = np.array(features, dtype=np.float32).reshape(1, -1)
        x = scaler.transform(x).reshape(1, 1, FEATURE_COUNT)

        probs = model.predict(x, verbose=0)[0]
        allowed_indices = ALPHABET_INDICES if current_mode == "ALPHABET" else GESTURE_INDICES

        best_allowed_index = allowed_indices[int(np.argmax(probs[allowed_indices]))]
        predicted_label = label_encoder.inverse_transform([best_allowed_index])[0]
        confidence = float(probs[best_allowed_index])

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_buffer.append(predicted_label)

            stable_prediction, stable_count = Counter(prediction_buffer).most_common(1)[0]
            display_prediction = format_prediction(stable_prediction)

            if stable_count >= MIN_STABLE_VOTES:
                if stable_prediction != candidate_prediction:
                    candidate_prediction = stable_prediction
                    candidate_since = now
                    candidate_committed = False

                if candidate_committed:
                    hold_status = "READY"
                else:
                    hold_elapsed = now - candidate_since
                    hold_status = f"{min(hold_elapsed, PREDICTION_DELAY):.1f}/{PREDICTION_DELAY:.1f}s"

                    if hold_elapsed >= PREDICTION_DELAY:
                        if current_mode == "ALPHABET":
                            current_word += stable_prediction.lower()
                        else:
                            sentence.append(stable_prediction)

                        candidate_committed = True
                        current_prediction = display_prediction
                        hold_status = "ADDED"
            else:
                candidate_prediction = ""
                candidate_since = 0.0
                candidate_committed = False
                hold_status = f"STABLE {stable_count}/{MIN_STABLE_VOTES}"
        else:
            display_prediction = "UNCERTAIN"
            prediction_buffer.clear()
            candidate_prediction = ""
            candidate_since = 0.0
            candidate_committed = False
    else:
        if now - last_hand_time > BUFFER_RESET_DELAY:
            prediction_buffer.clear()
            candidate_prediction = ""
            candidate_since = 0.0
            candidate_committed = False
            hold_status = f"{PREDICTION_DELAY:.1f}s"

    current_prediction = display_prediction

    active_word = current_word if current_mode == "ALPHABET" else ""
    saved_sentence_text = compose_saved_sentence_text(sentence)
    paragraph_text = compose_paragraph_text(paragraph)
    predicted_text = current_prediction
    building_word_text = current_word if current_mode == "ALPHABET" else ""

    fps = 1 / (time.time() - start_time + 1e-6)

    cv2.putText(ui, f"MODE : {current_mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(ui, f"Prediction : {current_prediction}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(ui, f"Confidence : {confidence:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(ui, f"Hold : {hold_status}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
    cv2.putText(ui, f"FPS : {fps:.1f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.rectangle(ui, (20, 285), (400, 395), (80, 80, 80), 2)
    cv2.rectangle(ui, (20, 415), (400, 585), (80, 80, 80), 2)
    cv2.rectangle(ui, (20, 605), (400, 785), (80, 80, 80), 2)

    cv2.putText(ui, "Predicted Alphabet/Word", (30, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    draw_wrapped_text(
        ui,
        predicted_text,
        (30, 345),
        340,
        30,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv2.putText(ui, "Current Word", (30, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 120), 1)
    draw_wrapped_text(
        ui,
        building_word_text if building_word_text else "-",
        (170, 375),
        190,
        28,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(ui, "Sentence (Saved Words)", (30, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    draw_wrapped_text(
        ui,
        saved_sentence_text if saved_sentence_text else "-",
        (30, 475),
        340,
        30,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    cv2.putText(ui, "Paragraph (Saved Sentences)", (30, 632), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    draw_wrapped_text(
        ui,
        paragraph_text if paragraph_text else "-",
        (30, 665),
        340,
        28,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )

    cv2.putText(
        ui,
        "A:Alphabet G:Gesture SPACE:Word P:Paragraph",
        (20, 820),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        ui,
        "V:Speak D:Delete C:Clear F/W:Screen Q:Quit",
        (20, 850),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    cv2.imshow(WINDOW_NAME, ui)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        current_mode = "ALPHABET"
        current_prediction = ""
        prediction_buffer.clear()
        candidate_prediction = ""
        candidate_since = 0.0
        candidate_committed = False

    elif key == ord("g"):
        if current_word.strip():
            sentence.append(current_word.strip())
            current_word = ""

        current_mode = "GESTURE"
        current_prediction = ""
        prediction_buffer.clear()
        candidate_prediction = ""
        candidate_since = 0.0
        candidate_committed = False

    elif key == ord(" "):
        if current_mode == "ALPHABET" and current_word.strip():
            sentence.append(current_word.strip())
            current_word = ""
            current_prediction = ""
            prediction_buffer.clear()
            candidate_prediction = ""
            candidate_since = 0.0
            candidate_committed = False

    elif key == ord("p"):
        if current_word.strip():
            sentence.append(current_word.strip())
            current_word = ""

        sentence_text = compose_saved_sentence_text(sentence)
        paragraph_sentence = format_sentence_for_paragraph(sentence_text)

        if paragraph_sentence:
            paragraph.append(paragraph_sentence)
            sentence.clear()
            current_prediction = ""
            prediction_buffer.clear()
            candidate_prediction = ""
            candidate_since = 0.0
            candidate_committed = False

    elif key == ord("v"):
        text_parts = []
        if paragraph:
            text_parts.append(compose_paragraph_text(paragraph))

        active_sentence_text = compose_sentence_text(
            sentence,
            current_word if current_mode == "ALPHABET" else "",
        )

        if active_sentence_text:
            text_parts.append(format_sentence_for_paragraph(active_sentence_text))

        text = " ".join(part for part in text_parts if part).strip()
        if text:
            speak_async(text)

    elif key == ord("d"):
        if current_mode == "ALPHABET" and current_word:
            current_word = current_word[:-1]
        elif sentence:
            sentence.pop()
        elif paragraph:
            paragraph.pop()

    elif key == ord("c"):
        sentence.clear()
        paragraph.clear()
        current_word = ""
        current_prediction = ""
        prediction_buffer.clear()
        candidate_prediction = ""
        candidate_since = 0.0
        candidate_committed = False

    elif key == ord("f"):
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif key == ord("w"):
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    elif key == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()

print("Application Closed Successfully")
