"""ASL sign recognizer - live webcam demo.

Run:   python app.py
Keys:  q / ESC = quit,  c = clear history,  l = toggle hand skeleton
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs

import argparse
import sys
import time
from collections import deque

import cv2

from utils import display
from utils.classifier import PredictionSmoother, load_classifier
from utils.config import MODEL_DIR, load_config
from utils.labels import pretty
from utils.preprocess import build_feature_vector

WINDOW_NAME = "ASL Sign Recognizer"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time ASL sign recognition from a webcam.")
    p.add_argument("--camera", type=int, default=None, help="camera index (default: from config, usually 0)")
    p.add_argument("--demo", action="store_true", help="force the rule-based demo mode (no trained model)")
    p.add_argument("--model-dir", default=str(MODEL_DIR), help="folder containing the model files")
    return p.parse_args()


def open_camera(index, width, height):
    """Open the webcam; return None (with a helpful message) on failure."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"\n[error] Could not open camera #{index}.")
        print("  - Is another app (Zoom, Teams, browser) using the webcam?")
        print("  - Try another index:  python app.py --camera 1")
        print("  - macOS: allow camera access for your terminal in System Settings > Privacy.")
        print("  - Windows: Settings > Privacy > Camera must allow desktop apps.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    args = parse_args()
    config = load_config()

    # Import here so a missing MediaPipe gives a friendly message, not a traceback.
    try:
        from utils.hand_detector import HandDetector
    except ImportError as err:
        print(f"[error] A required package is missing: {err}")
        print("        Run:  pip install -r requirements.txt")
        return 1

    from pathlib import Path
    classifier, notice = load_classifier(config, Path(args.model_dir), force_demo=args.demo)
    if notice:
        print(f"[info] {notice}")

    cam_cfg = config["camera"]
    cap = open_camera(args.camera if args.camera is not None else cam_cfg["index"],
                      cam_cfg["width"], cam_cfg["height"])
    if cap is None:
        return 1

    det_cfg = config["hand_detector"]
    detector = HandDetector(det_cfg["max_num_hands"], det_cfg["min_detection_confidence"],
                            det_cfg["min_tracking_confidence"], det_cfg["model_complexity"])
    smoother = PredictionSmoother(**config["smoothing"])

    history = deque(maxlen=5)    # last few recognised signs
    last_committed = None        # avoids adding the same sign to history every frame
    show_landmarks = True
    failed_reads = 0
    fps, prev_time = 0.0, time.time()

    print("Running. Press 'q' or ESC in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > 30:
                    print("[error] Lost the camera feed (30 failed reads in a row).")
                    return 1
                continue
            failed_reads = 0

            frame = cv2.flip(frame, 1)               # mirror view, like a selfie

            # 1) detect hands  2) build features  3) classify  4) smooth
            hands = detector.process(frame)
            features = build_feature_vector(hands)
            if features is None:
                raw_label, conf = None, 0.0
                last_committed = None                # hands left: allow repeating a sign
            else:
                raw_label, conf = classifier.predict(features, hands)
            label, conf = smoother.update(raw_label, conf)

            if label and label != last_committed:
                history.append(pretty(label))
                last_committed = label

            # ---- draw ----
            if show_landmarks:
                detector.draw(frame, hands)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            display.draw_header(frame, classifier.name, fps)
            if notice:
                display.draw_notice(frame, notice)
            if not hands:
                display.draw_hint(frame, "Show your hand to the camera")
            elif label:
                display.draw_prediction(frame, pretty(label), conf if classifier.name != "DEMO MODE (rule-based)" else None)
            else:
                display.draw_hint(frame, "...")
            display.draw_footer_help(frame, "q: quit   c: clear   l: toggle skeleton")
            display.draw_history(frame, list(history))

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                history.clear()
                smoother.reset()
            if key == ord("l"):
                show_landmarks = not show_landmarks
    except KeyboardInterrupt:
        pass
    finally:                                         # always release the camera
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
