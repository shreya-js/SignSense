"""SignSense live demo.

Controls:
    hold a sign steady for ~1.5s -> letter is added to the sentence
    SPACE (physical hand sign not needed) -> not used; space is auto via
        holding an open palm briefly is NOT implemented to keep this simple.
        Instead, use these keys on the keyboard:
    [space bar]  -> add a space to the sentence
    [backspace]  -> delete last character
    [s]          -> speak the sentence out loud (needs pyttsx3)
    [c]          -> clear the sentence
    [ESC]        -> quit

Usage:
    python live_predict.py
"""
import json
import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

from utils import normalize

# ---- Load model ----
model = keras.models.load_model("models/signsense_mlp.keras")
with open("models/labels.json") as f:
    labels = json.load(f)

# ---- Try to set up text-to-speech (optional) ----
try:
    import pyttsx3
    pyttsx3.init()  # just to confirm it's installed/working
    tts_available = True
except Exception:
    tts_available = False
    print("pyttsx3 not available - speak feature disabled. Install with: pip install pyttsx3")


def speak(text):
    """Create a fresh engine for every call.

    pyttsx3 on Windows (SAPI5) has a known issue where reusing one engine
    object across multiple runAndWait() calls causes later calls to go
    silent. Re-initializing each time avoids that.
    """
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("TTS error:", e)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

# Smoothing: majority vote over recent predictions
HISTORY_LEN = 10
CONFIDENCE_THRESHOLD = 0.75
HOLD_SECONDS = 1.5  # how long a stable letter must be held before it's typed

pred_history = deque(maxlen=HISTORY_LEN)
sentence = ""
stable_letter = None
stable_since = None
last_committed = None

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    display_letter = "-"
    confidence = 0.0

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        flat = [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)]
        feats = normalize(flat).reshape(1, -1)
        probs = model.predict(feats, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        raw_letter = labels[idx] if confidence >= CONFIDENCE_THRESHOLD else None
        pred_history.append(raw_letter)

        # majority vote over recent frames (ignoring None / low-confidence)
        votes = [p for p in pred_history if p is not None]
        if votes:
            display_letter, count = Counter(votes).most_common(1)[0]
            if count < HISTORY_LEN * 0.6:
                display_letter = "-"
        else:
            display_letter = "-"

        # hold-to-commit logic
        if display_letter != "-" :
            if display_letter == stable_letter:
                if stable_since and time.time() - stable_since >= HOLD_SECONDS:
                    if display_letter != last_committed:
                        sentence += " " if display_letter == "space" else display_letter
                        last_committed = display_letter
                        stable_since = time.time()  # reset so it doesn't repeat immediately
            else:
                stable_letter = display_letter
                stable_since = time.time()
                last_committed = None if display_letter != last_committed else last_committed
        else:
            stable_letter = None
            stable_since = None
            last_committed = None
    else:
        pred_history.clear()
        stable_letter = None
        stable_since = None
        last_committed = None
        cv2.putText(frame, "No hand detected", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ---- HUD ----
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    shown = "SPACE" if display_letter == "space" else display_letter.upper()
    cv2.putText(frame, f"Letter: {shown} ({confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # progress bar for hold-to-commit
    if stable_letter and stable_since:
        progress = min(1.0, (time.time() - stable_since) / HOLD_SECONDS)
        cv2.rectangle(frame, (10, 45), (10 + int(200 * progress), 60), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, 45), (210, 60), (255, 255, 255), 1)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE=space  BACKSPACE=del  S=speak  C=clear  ESC=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("SignSense - live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # space
        sentence += " "
    elif key == 8:  # backspace
        sentence = sentence[:-1]
    elif key in (ord("c"), ord("C")):
        sentence = ""
    elif key in (ord("s"), ord("S")):
        if tts_available and sentence.strip():
            speak(sentence)

cap.release()
cv2.destroyAllWindows()
print("Final sentence:", sentence)