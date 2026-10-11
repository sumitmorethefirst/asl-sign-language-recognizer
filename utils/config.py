"""Loads model/pretrained_model_config.json, falling back to safe defaults."""

import json
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
CONFIG_PATH = MODEL_DIR / "pretrained_model_config.json"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

DEFAULT_CONFIG = {
    "hand_detector": {
        "max_num_hands": 2,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
        "model_complexity": 0,
    },
    "classifier": {
        "model_file": "asl_model.keras",
        "labels_file": "labels.json",
        "confidence_threshold": 0.7,
    },
    "smoothing": {"window_size": 10, "min_votes": 6},
    "camera": {"index": 0, "width": 960, "height": 540},
}


def load_config(path=CONFIG_PATH):
    """Return the config dict. Never raises: problems become warnings."""
    config = deepcopy(DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
    except FileNotFoundError:
        print(f"[config] {path} not found - using built-in defaults.")
        return config
    except json.JSONDecodeError as err:
        print(f"[config] {path} is not valid JSON ({err}) - using defaults.")
        return config

    for section, values in user_config.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    return config
