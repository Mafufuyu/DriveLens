#pragma once

// ── Server ────────────────────────────────────────────────────────────
constexpr const char* API_ENDPOINT         = "http://localhost:8000/upload";
constexpr int         UPLOAD_TIMEOUT_MS    = 30000;

// ── Capture ───────────────────────────────────────────────────────────
constexpr int         CAPTURE_INTERVAL_SEC = 2;

// ── Image ─────────────────────────────────────────────────────────────
constexpr int         RESIZE_WIDTH         = 640;
constexpr int         RESIZE_HEIGHT        = 480;
constexpr int         JPEG_QUALITY         = 80;

// ── Debug ─────────────────────────────────────────────────────────────
#define DEBUG_SAVE_FRAMES
constexpr const char* DEBUG_OUTPUT_DIR     = "debug_frames";
