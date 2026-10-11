"""Thin wrapper around MediaPipe Hands (the pretrained part of this project).

MediaPipe Hands is a pair of neural networks trained by Google:
  1. a palm detector that finds hands in the image, and
  2. a landmark model that predicts 21 3-D keypoints per hand.
We only use it as a feature extractor.
"""

from dataclasses import dataclass, field
from typing import Any, List

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandResult:
    landmarks: np.ndarray      # shape (21, 3): x, y (0-1 image coords), z (relative depth)
    label: str                 # "Left" or "Right"
    score: float               # handedness confidence
    raw: Any = field(default=None, repr=False)  # MediaPipe object, used for drawing


class HandDetector:
    def __init__(self, max_hands=2, detection_conf=0.6, tracking_conf=0.5, model_complexity=0):
        self._mp_hands = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,       # video mode: track hands between frames
            max_num_hands=max_hands,
            model_complexity=model_complexity,  # 0 = fastest, good for laptops
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame_bgr) -> List[HandResult]:
        """Detect hands in a BGR frame. Returns [] if none are found.

        Tip: flip the frame horizontally (mirror) BEFORE calling this so that
        the "Left"/"Right" labels match the signer's real hands.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
        rgb.flags.writeable = False                         # small speed-up
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        hands = []
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            top = handedness.classification[0]
            hands.append(HandResult(coords, top.label, float(top.score), raw=lm))
        return hands

    def draw(self, frame_bgr, hands):
        """Draw the skeleton of every detected hand onto the frame (in place)."""
        for hand in hands:
            self._drawing.draw_landmarks(frame_bgr, hand.raw, self._mp_hands.HAND_CONNECTIONS)

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
