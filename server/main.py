"""
DriveLens Cloud Server
──────────────────────
Receives full JPEG frames from the C++ edge agent, runs YOLOv8 object
detection locally, and stores results in SQLite.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Model is downloaded automatically on first run:
    - YOLOv8n  (~6 MB, saved to working directory)
"""

from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException

from database import init_db, insert_detection, get_all_detections
from ocr import analyze_image, load_models

# ── Debug switch ─────────────────────────────────────────────────────
DEBUG: bool = True

# ── Configuration ─────────────────────────────────────────────────────
RECEIVED_DIR = Path("received_images")

app = FastAPI(title="DriveLens Cloud Server")


@app.on_event("startup")
def startup():
    """Create output directory, initialize the database, and pre-load AI model."""
    RECEIVED_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    load_models()
    print(f"[Server] Saving images to: {RECEIVED_DIR.resolve()}")


@app.post("/upload")
async def upload_frame(file: UploadFile = File(...)):
    """
    Receive a JPEG frame from the C++ edge client.

    Pipeline:
        1. Save image to disk
        2. YOLOv8 object detection
        3. Store results in SQLite
        4. Return JSON to C++ client
    """
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received.")

        # --- 1. Save image to disk ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = file.filename if file.filename else f"frame_{timestamp}.jpg"
        save_path = RECEIVED_DIR / filename
        save_path.write_bytes(contents)

        size_kb = len(contents) / 1024
        print(f"\n{'='*60}")
        print(f"Received image: {filename} ({size_kb:.1f} KB)")

        # --- 2. YOLOv8 object detection ---
        vision_result = analyze_image(contents)

        detected_objects = vision_result["objects"]
        image_width      = vision_result["image_width"]
        image_height     = vision_result["image_height"]

        if DEBUG:
            if detected_objects:
                for obj in detected_objects:
                    print(f"[OBJECT]  {obj['name']:<20s}  "
                          f"confidence={obj['confidence']:.1%}  "
                          f"bbox=({obj['x_min']},{obj['y_min']})-"
                          f"({obj['x_max']},{obj['y_max']})")
            else:
                print(f"[OBJECT]  (no objects detected)")

        # --- 3. Save to database ---
        row_id = insert_detection(filename, detected_objects)
        print(f"[DB] Saved detection #{row_id}")
        if DEBUG:
            print(f"{'='*60}")

        # --- 4. Return JSON to C++ client ---
        return {
            "status": "ok",
            "filename": filename,
            "size_bytes": len(contents),
            "image_width": image_width,
            "image_height": image_height,
            "detected_objects": detected_objects,
            "db_id": row_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Failed to process upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detections")
def list_detections():
    """Return all stored detection records (newest first)."""
    return get_all_detections()


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "running"}
