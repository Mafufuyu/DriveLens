"""
DriveLens Cloud Server
──────────────────────
Receives JPEG frames uploaded by the DriveLens C++ edge agent.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException

# ── Configuration ─────────────────────────────────────────────────────
RECEIVED_DIR = Path("received_images")

app = FastAPI(title="DriveLens Cloud Server")


@app.on_event("startup")
def startup():
    """Create the output directory on server start."""
    RECEIVED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Server] Saving images to: {RECEIVED_DIR.resolve()}")


@app.post("/upload")
async def upload_frame(file: UploadFile = File(...)):
    """
    Receive a JPEG frame from the C++ client.

    The client sends a multipart/form-data POST with a field named "file".
    """
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received.")

        # Build a unique filename: original name or timestamp-based fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = file.filename if file.filename else f"frame_{timestamp}.jpg"
        save_path = RECEIVED_DIR / filename

        save_path.write_bytes(contents)

        size_kb = len(contents) / 1024
        print(f"Image received: {filename} ({size_kb:.1f} KB)")

        return {"status": "ok", "filename": filename, "size_bytes": len(contents)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Failed to process upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "running"}
