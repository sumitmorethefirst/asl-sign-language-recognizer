# data/

- `labels.txt`  – the sign vocabulary (edit it to add/remove signs).
- `samples/`    – one CSV per sign (e.g. `hello.csv`), created by
  `python scripts/collect_data.py`. Each row is one frame of hand-landmark
  features (129 numbers). No images or video are stored, only numbers.

CSV files are git-ignored by default.
