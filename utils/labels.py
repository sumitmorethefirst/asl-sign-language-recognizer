"""Sign vocabulary helpers."""

from pathlib import Path

from utils.config import PROJECT_ROOT

LABELS_FILE = PROJECT_ROOT / "data" / "labels.txt"

# Used only if data/labels.txt is missing or empty.
DEFAULT_LABELS = [
    "hello", "yes", "no", "thank_you", "please",
    "help", "family", "friend", "eat", "drink",
]


def load_labels(path=LABELS_FILE):
    """Read the vocabulary from a text file (one label per line)."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"[labels] {path} not found - using default vocabulary.")
        return list(DEFAULT_LABELS)

    labels = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return labels or list(DEFAULT_LABELS)


def pretty(label):
    """'thank_you' -> 'THANK YOU' (nice for on-screen text)."""
    return label.replace("_", " ").upper()
