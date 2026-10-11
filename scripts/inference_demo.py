"""Run the recognizer on a single image (no webcam needed).

    python scripts/inference_demo.py path/to/photo.jpg [--demo]

Useful for testing the pipeline. The image is mirrored first, because the live
app mirrors the webcam; this keeps Left/Right hand labels consistent.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from utils.classifier import load_classifier
from utils.config import load_config
from utils.labels import pretty
from utils.preprocess import build_feature_vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--demo", action="store_true", help="use rule-based demo mode")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"[error] Could not read image: {args.image}")
        return 1

    try:
        from utils.hand_detector import HandDetector
    except ImportError as err:
        print(f"[error] Missing package: {err}. Run: pip install -r requirements.txt")
        return 1

    config = load_config()
    classifier, notice = load_classifier(config, force_demo=args.demo)
    if notice:
        print(f"[info] {notice}")

    with HandDetector() as detector:
        hands = detector.process(cv2.flip(image, 1))
    features = build_feature_vector(hands)
    if features is None:
        print("No hand detected in the image.")
        return 0

    label, conf = classifier.predict(features, hands)
    if label is None:
        print(f"Hand found, but no confident match (confidence {conf:.2f}).")
    else:
        print(f"Prediction: {pretty(label)}  (confidence {conf:.2f}, mode: {classifier.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
