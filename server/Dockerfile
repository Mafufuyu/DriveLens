# ── Stage: Runtime ────────────────────────────────────────────────────
FROM python:3.10-slim

# System libraries required by OpenCV and Pillow on slim base
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch (CPU-only, ~250 MB vs ~2 GB CUDA build) ───────────
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ─────────────────────────────
# Copy requirements first so this layer is cached unless deps change
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    "ultralytics>=8.2.0" \
    Pillow

# ── Pre-download YOLOv8n weights at build time ────────────────────────
# Avoids the download delay on first request at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Copy application source ───────────────────────────────────────────
COPY . .

# Persistent storage for received images and the SQLite database
VOLUME ["/app/received_images", "/app/drivelens.db"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
