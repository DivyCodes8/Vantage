"""File-backed RoomState store for Vantage's perception layer.

Holds everything the camera has seen — object label, where it is (left / center /
right + bounding box), how confident the detector was, and first/last-seen
timestamps — in a single JSON file. Writes are ATOMIC (temp file + rename) so a
concurrent reader (the voice agent, later) never observes a half-written file.

Design notes
------------
* One entry per object label. When the same label is seen again we UPDATE its
  position + confidence + last_seen rather than appending a duplicate. If several
  instances of a label appear in one frame, the highest-confidence one wins.
* Objects that are no longer visible are NOT deleted — they stay in memory with
  their old ``last_seen`` and ``visible=False`` so the agent can answer
  "where was my charger?" even after it leaves frame.
* Detections below ``CONFIDENCE_THRESHOLD`` are still stored, but flagged
  ``low_confidence=True`` so the agent can hedge ("I think I saw ... not sure").
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Detections at or above this confidence are treated as reliable. Anything below
# is still remembered, just marked low_confidence so the agent can soften it.
CONFIDENCE_THRESHOLD = 0.45

# room_state.json lives next to this module by default.
DEFAULT_PATH = Path(__file__).resolve().parent / "room_state.json"


def region_of_bbox(bbox: Iterable[float], frame_width: float) -> str:
    """Map a bbox to a coarse horizontal region: left / center / right.

    Uses the bbox center-x split into thirds of the frame width.
    """
    x1, _y1, x2, _y2 = bbox
    cx = (float(x1) + float(x2)) / 2.0
    third = frame_width / 3.0
    if cx < third:
        return "left"
    if cx < 2.0 * third:
        return "center"
    return "right"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


class RoomState:
    """A small JSON-backed store of what the room currently/recently contained."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self._lock = threading.RLock()
        # label -> entry dict
        self._objects: dict[str, dict[str, Any]] = {}
        self._caption: str = ""
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt/partial file — start fresh rather than crash.
            return
        self._caption = data.get("scene_caption", "")
        for entry in data.get("objects", []):
            label = entry.get("label")
            if label:
                self._objects[label] = entry

    # ---------------------------------------------------------------- update
    def upsert(self, detections: Iterable[dict], frame_width: float | None = None) -> None:
        """Merge a batch of detections from one frame into the store.

        Each detection is a dict with ``label``, ``bbox`` ([x1, y1, x2, y2]) and
        ``confidence``. ``frame_width`` lets us compute the left/center/right
        region; if omitted, a detection may carry a precomputed ``region``.
        """
        now = time.time()

        # Collapse to the best detection per label for this frame.
        best: dict[str, dict] = {}
        for d in detections:
            label = d["label"]
            if label not in best or float(d["confidence"]) > float(best[label]["confidence"]):
                best[label] = d

        with self._lock:
            seen_now: set[str] = set()
            for label, d in best.items():
                seen_now.add(label)
                conf = float(d["confidence"])
                bbox = [float(v) for v in d["bbox"]]
                region = d.get("region")
                if region is None and frame_width:
                    region = region_of_bbox(bbox, frame_width)

                entry = self._objects.get(label)
                if entry is None:
                    entry = {"label": label, "first_seen": now, "first_seen_iso": _iso(now)}
                    self._objects[label] = entry

                entry.update(
                    {
                        "region": region,
                        "bbox": bbox,
                        "confidence": round(conf, 4),
                        "low_confidence": conf < CONFIDENCE_THRESHOLD,
                        "last_seen": now,
                        "last_seen_iso": _iso(now),
                        "visible": True,
                    }
                )

            # Anything not seen this frame stays in memory but is no longer visible.
            for label, entry in self._objects.items():
                if label not in seen_now:
                    entry["visible"] = False

            self._save()

    def set_caption(self, caption: str) -> None:
        """Store the latest scene caption (from Florence-2)."""
        with self._lock:
            self._caption = caption or ""
            self._save()

    def get_caption(self) -> str:
        with self._lock:
            return self._caption

    # ----------------------------------------------------------------- query
    def get_all(self) -> list[dict]:
        """Return a copy of all remembered objects."""
        with self._lock:
            return [dict(e) for e in self._objects.values()]

    def query(self, label: str) -> list[dict]:
        """Simple case-insensitive substring match on object labels.

        Returns matching entries sorted by confidence (highest first).
        """
        q = label.strip().lower()
        with self._lock:
            matches = [
                dict(e)
                for lbl, e in self._objects.items()
                if q in lbl.lower() or lbl.lower() in q
            ]
        matches.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
        return matches

    # ------------------------------------------------------------------ save
    def _save(self) -> None:
        """Atomically write the whole store to disk (temp file + rename)."""
        payload = {
            "updated": time.time(),
            "updated_iso": _iso(time.time()),
            "scene_caption": self._caption,
            "objects": list(self._objects.values()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".room_state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)  # atomic on POSIX
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


if __name__ == "__main__":
    # Tiny self-test: no camera needed.
    import tempfile as _tf

    p = Path(_tf.gettempdir()) / "vantage_room_state_selftest.json"
    if p.exists():
        p.unlink()
    rs = RoomState(p)
    rs.upsert(
        [
            {"label": "cup", "bbox": [10, 10, 60, 80], "confidence": 0.91},
            {"label": "laptop", "bbox": [400, 100, 600, 300], "confidence": 0.3},
        ],
        frame_width=640,
    )
    rs.upsert([{"label": "cup", "bbox": [500, 10, 560, 80], "confidence": 0.95}], frame_width=640)
    print("all:", json.dumps(rs.get_all(), indent=2))
    print("query 'charge':", rs.query("charge"))
    print("query 'cup':", [(e["label"], e["region"], e["visible"]) for e in rs.query("cup")])
