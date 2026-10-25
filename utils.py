import numpy as np

# 24 static ASL letters (J and Z need motion, so they are excluded)
LETTERS = list("abcdefghiklmnopqrstuvwxy")


def normalize(flat_landmarks):
    """Turn 63 raw MediaPipe values (21 points x,y,z) into a position- and
    scale-independent feature vector.

    1. Shift so the wrist (point 0) is the origin.
    2. Divide by the largest distance from the wrist, so hand size and
       distance from the camera stop mattering.
    """
    pts = np.asarray(flat_landmarks, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()


def landmarks_to_flat(hand_landmarks):
    """Convert a MediaPipe hand result into a flat list of 63 numbers."""
    return [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
