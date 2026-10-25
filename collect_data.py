"""Collect hand-landmark samples for each letter.

Usage:
    python collect_data.py --person shreya
    python collect_data.py --person friend1 --samples 250

Controls:
    press a letter key (a-y, no j) -> start recording that letter
    ,  (comma)                     -> start recording the "space" gesture
                                       (open palm, fingers SPREAD apart --
                                       different from letter B, which is
                                       fingers held together)
    SPACE BAR                      -> pause recording
    ESC                            -> quit

Tip: while a letter is recording, slowly move/tilt your hand a little and
change the distance to the camera. This variety makes the model robust.
"""
import argparse
import csv
import os
import time

import cv2
import mediapipe as mp

from utils import LETTERS, landmarks_to_flat

parser = argparse.ArgumentParser()
parser.add_argument("--person", default="p1", help="name of the person signing")
parser.add_argument("--samples", type=int, default=200, help="samples per letter")
parser.add_argument("--out", default="data/landmarks.csv")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
new_file = not os.path.exists(args.out)
f = open(args.out, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow(["label", "person"] + [f"{a}{i}" for i in range(21) for a in "xyz"])

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
current = None
collected = 0
last_save = 0.0
done = {}

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror view
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        now = time.time()
        if current and collected < args.samples and now - last_save > 0.05:
            writer.writerow([current, args.person] + landmarks_to_flat(hand))
            collected += 1
            last_save = now
            if collected == args.samples:
                done[current] = collected
                f.flush()
                current = None
    else:
        cv2.putText(frame, "No hand detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    status = f"Recording '{current.upper()}': {collected}/{args.samples}" if current else "Press a letter key to record"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Done: {' '.join(sorted(k.upper() for k in done))}",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("SignSense - data collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        current = None
    elif key == ord(","):
        current = "space"
        collected = 0
    elif key != 255 and chr(key).lower() in LETTERS:
        current = chr(key).lower()
        collected = 0

f.close()
cap.release()
cv2.destroyAllWindows()
print("Saved to", args.out)