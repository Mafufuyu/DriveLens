"""
ocr.py – Local AI: YOLOv8 object detection.

Runs entirely on CPU – no cloud API, no GPU required.

Model is loaded once at startup and reused for every request:
    - YOLOv8n  : COCO nano model (~6 MB), auto-downloaded on first run
"""

import io
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── Driving-relevant COCO class IDs ──────────────────────────────────
_RELEVANT_CLASS_IDS = {0, 1, 2, 3, 5, 7, 9, 11}
_COCO_NAMES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "sign",
}

# ── Singleton model instance (loaded once, reused for every request) ──
_yolo_model: YOLO | None = None


def load_models() -> None:
    """Pre-load YOLO model. Called once at server startup."""
    global _yolo_model

    print("[AI] Loading YOLOv8n model (CPU)...")
    _yolo_model = YOLO("yolov8n.pt")

    print("[AI] Model ready.")


def _get_model() -> YOLO:
    """Return loaded model, lazy-loading if startup was skipped."""
    if _yolo_model is None:
        load_models()
    return _yolo_model  # type: ignore[return-value]


def _bytes_to_np(image_bytes: bytes) -> np.ndarray:
    """Convert raw JPEG bytes to an RGB numpy array."""
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def analyze_image(image_bytes: bytes) -> dict:
    """
    Run YOLOv8 on the given image bytes.

    Returns:
        {
            "image_width": 640,
            "image_height": 480,
            "objects": [
                {"name": "car", "confidence": 0.91,
                 "x_min": 120, "y_min": 45, "x_max": 310, "y_max": 220},
                ...
            ]
        }
    Coordinates are in pixels relative to the analyzed image.
    """
    yolo = _get_model()
    image_np = _bytes_to_np(image_bytes)
    h, w = image_np.shape[:2]

    # ── YOLO Object Detection ─────────────────────────────────────────
    detected_objects = []
    try:
        results = yolo(image_np, device="cpu", verbose=False)
        if results:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in _RELEVANT_CLASS_IDS:
                    continue

                conf = float(box.conf[0])
                if conf < 0.40:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detected_objects.append({
                    "name":       _COCO_NAMES.get(cls_id, "unknown"),
                    "confidence": round(conf, 3),
                    "x_min":      round(x1),
                    "y_min":      round(y1),
                    "x_max":      round(x2),
                    "y_max":      round(y2),
                })
    except Exception as e:
        print(f"[YOLO] Warning: {e}")

    return {
        "image_width":  w,
        "image_height": h,
        "objects":      detected_objects,
    }
