"""Record training samples for your own sign vocabulary.

Run from the project root:   python scripts/collect_data.py

How it works
  1. Choose a sign with the number keys (1-9, 0) or n / p (next / previous).
  2. Get into position, press SPACE to start recording, SPACE again to stop.
     While recording, ~10 samples per second are saved to data/samples/<sign>.csv.
  3. Vary your hand position, angle and distance a little while recording.
  4. Aim for 150+ samples per sign, then run scripts/train_or_prepare_model.py.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import utils`

import cv2
import numpy as np

from utils import display
from utils.config import SAMPLES_DIR, load_config
from utils.labels import load_labels, pretty
from utils.preprocess import build_feature_vector

SAVE_INTERVAL = 0.1  # seconds between saved samples


def count_samples(path):
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)


def main():
    p = argparse.ArgumentParser(description="Collect hand-landmark samples for each sign.")
    p.add_argument("--camera", type=int, default=None)
    args = p.parse_args()

    try:
        from utils.hand_detector import HandDetector
    except ImportError as err:
        print(f"[error] Missing package: {err}. Run: pip install -r requirements.txt")
        return 1

    config = load_config()
    labels = load_labels()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    counts = {lbl: count_samples(SAMPLES_DIR / f"{lbl}.csv") for lbl in labels}

    cam_index = args.camera if args.camera is not None else config["camera"]["index"]
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[error] Could not open camera #{cam_index}. Try --camera 1, or close other apps using it.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    det = config["hand_detector"]
    detector = HandDetector(det["max_num_hands"], det["min_detection_confidence"],
                            det["min_tracking_confidence"], det["model_complexity"])

    current, recording, last_save = 0, False, 0.0
    print("Keys: 1-9,0 pick sign | n/p next/prev | SPACE record on/off | q quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[error] Camera read failed.")
                break
            frame = cv2.flip(frame, 1)
            hands = detector.process(frame)
            features = build_feature_vector(hands)
            detector.draw(frame, hands)

            label = labels[current]
            if recording and features is not None and time.time() - last_save >= SAVE_INTERVAL:
                with open(SAMPLES_DIR / f"{label}.csv", "a") as f:
                    np.savetxt(f, features[None, :], delimiter=",", fmt="%.6f")
                counts[label] += 1
                last_save = time.time()

            status = "RECORDING" if recording else "paused"
            display.draw_text_with_bg(frame, f"Sign: {pretty(label)}  [{counts[label]} samples]  {status}",
                                      (15, 35), 0.7, (0, 0, 255) if recording else (255, 255, 255))
            if recording and features is None:
                display.draw_notice(frame, "No hand detected - nothing is being saved.")
            display.draw_footer_help(frame, "1-9,0 / n / p: choose sign   SPACE: record on/off   q: quit")
            # Small list of all signs with counts
            y = 80
            for i, lbl in enumerate(labels):
                mark = ">" if i == current else " "
                y += display.draw_text_with_bg(frame, f"{mark}{(i + 1) % 10} {pretty(lbl)}: {counts[lbl]}",
                                               (15, y), 0.45, (200, 200, 200), 1, pad=2)

            cv2.imshow("Collect ASL data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                recording = not recording
            elif key == ord("n"):
                current, recording = (current + 1) % len(labels), False
            elif key == ord("p"):
                current, recording = (current - 1) % len(labels), False
            elif ord("0") <= key <= ord("9"):
                idx = (key - ord("0") - 1) % 10        # '1'->0 ... '9'->8, '0'->9
                if idx < len(labels):
                    current, recording = idx, False
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
    print("Sample counts:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
