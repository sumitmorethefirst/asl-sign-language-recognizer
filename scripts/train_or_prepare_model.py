"""Train the classifier head on your recorded landmark samples.

Run from the project root:
    python scripts/train_or_prepare_model.py --check     # verify environment + data
    python scripts/train_or_prepare_model.py             # train and save the model

Transfer-learning idea: MediaPipe Hands is the pretrained "backbone" (it turns
pixels into 21 landmarks). Here we only train a SMALL classifier on top of its
output, which is why a few hundred samples per sign and a few seconds on a CPU
are enough.

Outputs:  model/asl_model.keras   and   model/labels.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.config import MODEL_DIR, SAMPLES_DIR, load_config
from utils.labels import load_labels
from utils.preprocess import FEATURE_SIZE

MIN_SAMPLES_PER_SIGN = 20


def load_dataset(labels):
    """Read data/samples/<label>.csv for each label. Returns X, y, used_labels."""
    X, y, used = [], [], []
    for label in labels:
        path = SAMPLES_DIR / f"{label}.csv"
        if not path.exists():
            print(f"  [skip] {label:<10} no data file")
            continue
        data = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float32)
        if data.shape[1] != FEATURE_SIZE:
            print(f"  [skip] {label:<10} wrong feature size {data.shape[1]} (expected {FEATURE_SIZE})")
            continue
        if len(data) < MIN_SAMPLES_PER_SIGN:
            print(f"  [skip] {label:<10} only {len(data)} samples (need {MIN_SAMPLES_PER_SIGN}+)")
            continue
        print(f"  [ok]   {label:<10} {len(data)} samples")
        X.append(data)
        y.extend([len(used)] * len(data))
        used.append(label)
    if not X:
        return None, None, []
    return np.vstack(X), np.array(y), used


def check_environment():
    for name in ("cv2", "mediapipe", "tensorflow", "sklearn"):
        try:
            module = __import__(name)
            print(f"  [ok]   {name} {getattr(module, '__version__', '')}")
        except ImportError:
            print(f"  [MISSING] {name}  ->  pip install -r requirements.txt")


def build_model(num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(FEATURE_SIZE,)),
        layers.GaussianNoise(0.02),               # light augmentation (only active in training)
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    p = argparse.ArgumentParser(description="Train the ASL landmark classifier.")
    p.add_argument("--check", action="store_true", help="only check packages and recorded data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    print("Environment:")
    check_environment()
    print("\nRecorded data:")
    labels = load_labels()
    X, y, used = load_dataset(labels)
    if args.check:
        return 0

    if X is None or len(used) < 2:
        print("\n[error] Need at least 2 signs with enough samples. Run scripts/collect_data.py first.")
        return 1

    try:
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from tensorflow import keras
    except ImportError as err:
        print(f"\n[error] {err}. Run: pip install -r requirements.txt")
        return 1

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_model(len(used))
    model.summary()

    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch_size, callbacks=[stop], verbose=2)

    preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nValidation report:")
    print(classification_report(y_val, preds, target_names=used, zero_division=0))
    print("Note: samples from the same recording are very similar, so this score is optimistic.\n"
          "Judge the model by trying it live with app.py.")

    cfg = load_config()["classifier"]
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / cfg["model_file"])
    with open(MODEL_DIR / cfg["labels_file"], "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2)
    print(f"Saved {MODEL_DIR / cfg['model_file']} and {MODEL_DIR / cfg['labels_file']}")
    print("Now run:  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
