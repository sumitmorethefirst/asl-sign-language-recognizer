<<<<<<< HEAD
# ASL Sign Recognizer (webcam demo)

A beginner-friendly Python project that watches your laptop webcam, detects your
hands with **MediaPipe**, classifies the hand shape with a small **TensorFlow**
model, and prints the recognised **ASL sign** on the video in real time.

It recognises **individual signs** (e.g. HELLO, YES, NO, THANK YOU, PLEASE, HELP,
FAMILY, FRIEND, EAT, DRINK) - not full sentences.

> ### Please read: what "pretrained" means here
> There is no public, ready-made model that recognises *these specific ASL words*
> from MediaPipe landmarks. So the project is split in two parts:
>
> | Part | Status |
> |---|---|
> | **MediaPipe Hands** (finds the hand + 21 landmarks) | **Pretrained by Google**, downloaded automatically with `pip` |
> | **Sign classifier** (landmarks -> sign name) | A *small* network you train in about 5 minutes on samples you record yourself (transfer-learning style: pretrained feature extractor + tiny trained head) |
>
> Until you train it, the app runs in **demo mode**: a rule-based recogniser for a
> few easy static hand shapes (open hand, fist, thumbs-up, "I love you", ...), so you can
> see the whole pipeline working immediately.

---

## 1. Requirements

- Python **3.10, 3.11 or 3.12** (64-bit)
- A webcam
- Any modern laptop; no GPU needed

## 2. Installation

```bash
git clone <your-repo-url> asl-sign-recognizer
cd asl-sign-recognizer

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Check that everything installed and see what data you have:

```bash
python scripts/train_or_prepare_model.py --check
```

## 3. Run it (demo mode - no training needed)

```bash
python app.py
```

Hold a hand up to the camera:

| Hand shape | Shown as |
|---|---|
| Open hand, all fingers spread | HELLO |
| Closed fist | YES |
| Thumb + index + middle finger out (or index + middle) | NO |
| Thumb, index and pinky out | I LOVE YOU |
| Thumbs up | GOOD |

Keys: `q` / `ESC` quit, `c` clear history, `l` toggle the hand skeleton.

Options: `python app.py --camera 1` (other webcam), `--demo` (force demo mode).

## 4. Model setup: train on the full vocabulary

Real ASL signs differ from person to person (and many involve movement), so you
teach the model *your* version of each sign. It is quick:

**Step 1 - Pick your vocabulary** (optional): edit `data/labels.txt`.

**Step 2 - Record samples**

```bash
python scripts/collect_data.py
```

- Choose a sign with `1`-`9`, `0` (or `n` / `p`).
- Press `SPACE` to start recording, `SPACE` to stop (about 10 samples per second).
- Record **150+ samples per sign**. Move your hands slightly, change distance
  and angle, and, if you want it to work for others, let friends record too.
- For each sign, record the **most distinctive moment** of the sign (the end
  handshape / position). See "Limitations" below.

**Step 3 - Train**

```bash
python scripts/train_or_prepare_model.py
```

This creates `model/asl_model.keras` and `model/labels.json`.

**Step 4 - Run**

```bash
python app.py
```

The header now says **TRAINED MODEL** and shows a confidence percentage.
Delete the two model files any time to go back to demo mode.

### Test on a single photo (no webcam)

```bash
python scripts/inference_demo.py path/to/hand.jpg
```

## 5. Folder structure

```
asl-sign-recognizer/
├── app.py                         # webcam demo (start here)
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── pretrained_model_config.json   # thresholds, MediaPipe + camera settings
│   ├── asl_model.keras                # created by training (git-ignored)
│   └── labels.json                    # created by training (git-ignored)
├── utils/
│   ├── hand_detector.py           # MediaPipe Hands wrapper
│   ├── preprocess.py              # landmarks -> normalised feature vector
│   ├── classifier.py              # trained model, demo fallback, smoothing
│   ├── labels.py                  # vocabulary loading
│   ├── display.py                 # on-screen text drawing
│   └── config.py                  # reads the JSON config
├── data/
│   ├── labels.txt                 # your sign vocabulary
│   └── samples/                   # recorded CSVs (git-ignored)
└── scripts/
    ├── collect_data.py            # record samples with the webcam
    ├── train_or_prepare_model.py  # check setup / train the classifier
    └── inference_demo.py          # classify one image file
```

## 6. How it works

```
 webcam frame ──► MediaPipe Hands ──► 21 landmarks/hand ──► normalise ──► 129 numbers
                  (pretrained)         (x, y, z each)        (preprocess.py)
                                                                     │
        text on screen ◄── majority vote over ◄── small neural net ◄─┘
                           last 10 frames        (Dense 128 → 64 → softmax)
```

1. **Hand landmark extraction.** MediaPipe Hands runs two pretrained neural
   networks: a palm detector, then a landmark model that outputs 21 3-D points
   (wrist, knuckles, fingertips...) per hand, in real time on a CPU.
2. **Feature processing** (`utils/preprocess.py`). Each hand is shifted so the
   wrist is the origin and scaled by hand size, so the result depends on the
   hand *shape*, not where it is or how far from the camera. Left and right
   hands get fixed slots (zeros if absent) plus the offset between the wrists,
   which helps with two-handed signs like HELP and FAMILY. Result: 129 numbers.
3. **Classification** (`utils/classifier.py`). A small fully-connected network
   maps those 129 numbers to a probability for each sign. Predictions below the
   confidence threshold (0.7) are ignored.
4. **Smoothing.** The sign is only shown when it wins a vote in at least 6 of the
   last 10 frames, so the text doesn't flicker.

**Why MediaPipe + a small classifier suits ASL**

- Landmarks remove the background, lighting, skin tone and clothing, which
  are the main things that break image classifiers.
- 129 numbers instead of thousands of pixels means a tiny model, a few hundred
  samples, seconds of training, and real-time speed on a normal laptop.
- The pretrained part (MediaPipe) already solved the hardest vision problem, so
  you only teach the model what *your* signs look like.

## 7. Limitations (honest notes)

- **Single-frame recognition.** The classifier looks at one moment at a time.
  Many ASL signs (HELLO, THANK YOU, YES, NO, EAT, DRINK...) include motion and
  face/body location, which a single hand pose can't fully capture. Signs that
  differ mainly by movement may be confused. Two-handed signs work best when
  both hands are in frame.
- **Hands only.** ASL also uses facial expression and body position; this
  project doesn't.
- **Demo-quality accuracy.** It is a learning/demo project, not an assistive
  communication tool. Don't rely on it for real-world communication.
- Training scores are optimistic (neighbouring frames are similar). Judge by
  using it live.
- Works best with good front lighting, a plain-ish background, and hands
  about 0.5-1 m from the camera.

**Ideas to extend it:** stack the features from the last ~30 frames and use an
LSTM/1-D CNN (captures motion); add MediaPipe Pose/Face landmarks; add a
text-to-speech call on each recognised sign.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Close Zoom/Teams/browser tabs using the webcam; try `--camera 1`; on macOS allow camera access for your terminal. |
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt`. |
| `pip` can't find mediapipe / tensorflow | Use Python 3.10-3.12 (64-bit). Newer Python versions may not be supported yet. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Install the pinned version: `pip install "mediapipe>=0.10.9,<0.11"`. |
| NumPy version conflict | `pip install "numpy<2"`. |
| Predictions flicker or are wrong | Record more varied samples; lower/raise `confidence_threshold` in `model/pretrained_model_config.json`. |
| App says "TensorFlow is not installed - using demo mode" | Demo mode still works; install TensorFlow to use a trained model. |
| Slow / low FPS | Lower `width`/`height` in the config; keep `model_complexity` at 0. |

## 9. Customising

All the main settings (camera size, MediaPipe confidence, smoothing window,
confidence threshold) live in `model/pretrained_model_config.json`.
To add a sign: add a line to `data/labels.txt`, record it, and re-train.
=======
# asl-sign-language-recognizer
Real-time American Sign Language recognition project using Python, MediaPipe, and a pretrained classifier model to detect individual signs and display text
>>>>>>> cede63caae06622195249e7bb45f4c0e0763e3bc
