"""Turn raw MediaPipe landmarks into a fixed-length feature vector.

Feature layout (129 numbers):
    [ left hand: 21*3 | right hand: 21*3 | wrist offset between hands: 3 ]

Each hand is normalised so the vector describes the hand SHAPE only:
  * translation: the wrist becomes the origin (hand position in frame is ignored)
  * scale: divided by the hand's size (distance from camera is ignored)
A missing hand is filled with zeros. The wrist offset lets the model tell apart
two-handed signs such as HELP or FAMILY.
"""

import numpy as np

NUM_LANDMARKS = 21
VALUES_PER_HAND = NUM_LANDMARKS * 3
FEATURE_SIZE = VALUES_PER_HAND * 2 + 3


def normalize_hand(landmarks):
    """(21, 3) landmarks -> (flat 63-vector, scale)."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    pts = pts - pts[0]                                  # wrist -> origin
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:                                    # degenerate detection
        scale = 1.0
    return (pts / scale).flatten(), scale


def build_feature_vector(hands):
    """Convert a list of HandResult objects into one feature vector.

    Returns None when no hands are visible.
    """
    if not hands:
        return None

    slots = [None, None]                                # [left, right]
    for hand in sorted(hands, key=lambda h: -h.score)[:2]:
        idx = 0 if hand.label == "Left" else 1
        if slots[idx] is not None:                      # both labelled the same? use the other slot
            idx = 1 - idx
        if slots[idx] is None:
            slots[idx] = hand

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    scales = []
    for i, hand in enumerate(slots):
        if hand is None:
            continue
        flat, scale = normalize_hand(hand.landmarks)
        features[i * VALUES_PER_HAND:(i + 1) * VALUES_PER_HAND] = flat
        scales.append(scale)

    if slots[0] is not None and slots[1] is not None:
        offset = slots[1].landmarks[0] - slots[0].landmarks[0]
        features[-3:] = offset / (sum(scales) / len(scales))
    return features
