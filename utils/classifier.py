"""Sign classifiers + prediction smoothing.

Two classifiers share the same interface  predict(features, hands) -> (label, confidence):

  * KerasSignClassifier  - your trained model (model/asl_model.keras)
  * HeuristicClassifier  - tiny rule-based fallback so the app runs out of the box

`load_classifier()` picks the right one and explains why.
"""

import json
from collections import Counter, deque

import numpy as np

from utils.config import MODEL_DIR
from utils.preprocess import FEATURE_SIZE


# --------------------------------------------------------------------------
# Trained model
# --------------------------------------------------------------------------
class KerasSignClassifier:
    name = "TRAINED MODEL"

    def __init__(self, model_path, labels_path, threshold=0.7):
        import tensorflow as tf  # imported lazily: heavy, and optional for demo mode

        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.threshold = threshold

        in_dim, out_dim = self.model.input_shape[-1], self.model.output_shape[-1]
        if in_dim != FEATURE_SIZE:
            raise ValueError(f"model expects {in_dim} features but the app produces {FEATURE_SIZE}. "
                             "Re-train with scripts/train_or_prepare_model.py.")
        if out_dim != len(self.labels):
            raise ValueError(f"model has {out_dim} outputs but labels.json has {len(self.labels)} labels.")

    def predict(self, features, hands=None):
        probs = self.model(features[None, :], training=False).numpy()[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        if confidence < self.threshold:
            return None, confidence
        return self.labels[best], confidence


# --------------------------------------------------------------------------
# Rule-based fallback (demo mode)
# --------------------------------------------------------------------------
def finger_states(landmarks):
    """Return (thumb, index, middle, ring, pinky) as 1 = extended, 0 = curled.

    Uses simple distances, so it works for any hand orientation.
    """
    lm = np.asarray(landmarks)

    def dist(a, b):
        return float(np.linalg.norm(lm[a][:2] - lm[b][:2]))

    # Thumb: the tip is far from the base of the pinky when the thumb is out.
    thumb = dist(4, 17) > dist(2, 17) * 1.15
    # Other fingers: the tip is farther from the wrist than the middle joint when extended.
    others = [dist(0, tip) > dist(0, pip) * 1.05
              for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    return tuple(int(v) for v in [thumb] + others)


class HeuristicClassifier:
    """Recognises a handful of STATIC hand shapes without any training.

    These are approximations for demo purposes - real ASL signs like HELLO or
    YES also involve movement. Train your own model for the full vocabulary.
    """
    name = "DEMO MODE (rule-based)"

    RULES = {
        (1, 1, 1, 1, 1): "hello",         # open flat hand
        (0, 0, 0, 0, 0): "yes",           # closed fist (the 'S' handshape used in YES)
        (1, 1, 1, 0, 0): "no",            # thumb + index + middle
        (0, 1, 1, 0, 0): "no",
        (1, 1, 0, 0, 1): "i_love_you",    # ASL 'ILY' handshape
        (1, 0, 0, 0, 0): "good",          # thumbs up
    }

    def predict(self, features, hands):
        if not hands:
            return None, 0.0
        label = self.RULES.get(finger_states(hands[0].landmarks))
        return (label, 1.0) if label else (None, 0.0)


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
def load_classifier(config, model_dir=MODEL_DIR, force_demo=False):
    """Return (classifier, notice). `notice` is None, or text explaining a fallback."""
    if force_demo:
        return HeuristicClassifier(), "Demo mode requested (--demo)."

    cfg = config["classifier"]
    model_path = model_dir / cfg["model_file"]
    labels_path = model_dir / cfg["labels_file"]

    if not model_path.exists() or not labels_path.exists():
        return HeuristicClassifier(), (
            "No trained model found - running rule-based demo. "
            "See README: collect_data.py then train_or_prepare_model.py.")
    try:
        clf = KerasSignClassifier(model_path, labels_path, cfg["confidence_threshold"])
        return clf, None
    except ImportError:
        return HeuristicClassifier(), "TensorFlow is not installed - using demo mode."
    except Exception as err:  # corrupt file, shape mismatch, etc.
        return HeuristicClassifier(), f"Could not load model ({err}) - using demo mode."


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------
class PredictionSmoother:
    """Majority vote over the last N frames so the text doesn't flicker."""

    def __init__(self, window_size=10, min_votes=6):
        self.min_votes = min_votes
        self._buffer = deque(maxlen=window_size)

    def update(self, label, confidence):
        self._buffer.append((label, confidence))
        counts = Counter(lbl for lbl, _ in self._buffer if lbl is not None)
        if not counts:
            return None, 0.0
        best, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return None, 0.0
        confs = [c for lbl, c in self._buffer if lbl == best]
        return best, float(np.mean(confs))

    def reset(self):
        self._buffer.clear()
