"""
database.py – SQLite storage for object detection results.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("drivelens.db")


def init_db():
    """Create the detections table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filename         TEXT    NOT NULL,
            detected_objects TEXT,
            timestamp        TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized database: {DB_PATH.resolve()}")


def insert_detection(filename: str,
                     detected_objects: list) -> int:
    """Insert a detection record and return its row id."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """
        INSERT INTO detections (filename, detected_objects, timestamp)
        VALUES (?, ?, ?)
        """,
        (filename,
         json.dumps(detected_objects, ensure_ascii=False),
         datetime.now().isoformat())
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_detections() -> list[dict]:
    """Return every detection record as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM detections ORDER BY id DESC").fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        if d.get("detected_objects"):
            d["detected_objects"] = json.loads(d["detected_objects"])
        results.append(d)
    return results
