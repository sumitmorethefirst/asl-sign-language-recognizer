"""Functions that draw text and overlays on the video frame (OpenCV)."""

import textwrap

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, YELLOW, GRAY = (80, 220, 80), (0, 220, 255), (200, 200, 200)


def draw_text_with_bg(frame, text, org, scale=0.7, color=WHITE, thickness=2, pad=6):
    """Draw text on a dark rectangle so it is readable on any background."""
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - pad, y - h - pad), (x + w + pad, y + baseline + pad), BLACK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return h + baseline + 2 * pad


def draw_header(frame, mode_name, fps):
    draw_text_with_bg(frame, f"{mode_name}   {fps:4.1f} FPS", (15, 30), 0.6, GRAY, 1)


def draw_notice(frame, message):
    """Show a (possibly long) warning/info message under the header."""
    y = 65
    for line in textwrap.wrap(message, width=80)[:3]:
        y += draw_text_with_bg(frame, line, (15, y), 0.5, YELLOW, 1)


def draw_history(frame, history):
    """Show the most recent recognised signs, bottom-left."""
    if history:
        text = "Recent: " + "  >  ".join(history)
        draw_text_with_bg(frame, text, (15, frame.shape[0] - 15), 0.6, GRAY, 1)


def draw_prediction(frame, label_text, confidence=None):
    """Big centred prediction near the bottom of the frame."""
    h, w = frame.shape[:2]
    text = label_text if confidence is None else f"{label_text} ({confidence * 100:.0f}%)"
    (tw, _), _ = cv2.getTextSize(text, FONT, 1.4, 3)
    draw_text_with_bg(frame, text, ((w - tw) // 2, h - 70), 1.4, GREEN, 3, pad=12)


def draw_hint(frame, text):
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    draw_text_with_bg(frame, text, ((w - tw) // 2, h - 70), 0.8, GRAY, 2)


def draw_footer_help(frame, text):
    draw_text_with_bg(frame, text, (15, frame.shape[0] - 45), 0.45, GRAY, 1, pad=3)
