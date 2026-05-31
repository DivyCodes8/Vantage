"""Scene definitions + builders for the deterministic fixtures.

Each scene is a list of objects ``(label, region, bbox, confidence, visible)``.
``build_state(name, now)`` turns a scene into a full room_state.json payload
with fresh timestamps so visible objects read as "just now".

Frame is treated as ~1920x1080. bboxes are chosen so check_hazards' spatial
heuristics fire as intended (e.g. the water bottle sits next to the laptop).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent

# (label, region, [x1,y1,x2,y2], confidence, visible)
SCENES: dict[str, dict] = {
    "desk": {
        "caption": "A desk with a laptop, a water bottle, a charger and a coffee cup.",
        "objects": [
            ("laptop",       "center", [600, 350, 950, 690],  0.93, True),
            ("water bottle", "center", [960, 300, 1060, 650], 0.90, True),  # next to laptop -> spill
            ("charger",      "right",  [1500, 420, 1610, 560], 0.91, True),
            ("cable",        "right",  [1430, 500, 1690, 565], 0.86, True),
            ("cup",          "left",   [150, 470, 300, 690],   0.88, True),
            ("desk",         "center", [0, 640, 1900, 1010],   0.95, True),  # surface for fall logic
        ],
    },
    "empty": {
        "caption": "A dark, mostly empty view — nothing clearly in frame.",
        "objects": [
            # Only low-confidence noise; nothing a person would call a real object.
            ("shadow", "center", [700, 400, 1200, 900], 0.22, True),
            ("blur",   "center", [300, 200, 800, 700],  0.18, True),
        ],
    },
    "changed": {
        # Same as desk, but the charger is GONE and a phone is NEW.
        "caption": "A desk with a laptop, a water bottle, a cup and a phone.",
        "objects": [
            ("laptop",       "center", [600, 350, 950, 690],  0.93, True),
            ("water bottle", "center", [960, 300, 1060, 650], 0.90, True),
            ("cable",        "right",  [1430, 500, 1690, 565], 0.86, True),
            ("cup",          "left",   [150, 470, 300, 690],   0.88, True),
            ("desk",         "center", [0, 640, 1900, 1010],   0.95, True),
            ("phone",        "center", [820, 500, 930, 720],   0.87, True),  # NEW
        ],
    },
}

CONFIDENCE_THRESHOLD = 0.45  # mirror room_state; keep fixtures self-contained


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def build_state(name: str, now: float | None = None) -> dict:
    """Build a full room_state payload for scene ``name`` with fresh timestamps."""
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; options: {list(SCENES)}")
    now = time.time() if now is None else now
    scene = SCENES[name]
    objects = []
    for label, region, bbox, conf, visible in scene["objects"]:
        # visible -> just seen; not visible -> a few minutes ago
        last_seen = now if visible else now - 200.0
        objects.append({
            "label": label,
            "first_seen": now - 300.0,
            "first_seen_iso": _iso(now - 300.0),
            "region": region,
            "bbox": [float(v) for v in bbox],
            "confidence": round(float(conf), 4),
            "low_confidence": conf < CONFIDENCE_THRESHOLD,
            "last_seen": last_seen,
            "last_seen_iso": _iso(last_seen),
            "visible": visible,
        })
    return {
        "updated": now,
        "updated_iso": _iso(now),
        "scene_caption": scene["caption"],
        "objects": objects,
    }


def baseline_snapshot(name: str, now: float | None = None) -> dict:
    """A change_snapshot payload (visible objects + regions) for scene ``name``.

    Used to seed what_changed()'s "earlier" baseline deterministically.
    """
    now = time.time() if now is None else now
    state = build_state(name, now)
    visible = {o["label"]: {"region": o["region"]} for o in state["objects"] if o["visible"]}
    return {"taken": now, "objects": visible}


def write_static_fixtures(now: float = 1_780_000_000.0) -> list[Path]:
    """Write fixture_<name>.json files (static, canonical scene definitions)."""
    written = []
    for name in SCENES:
        p = FIXTURE_DIR / f"fixture_{name}.json"
        p.write_text(json.dumps(build_state(name, now), indent=2))
        written.append(p)
    return written


if __name__ == "__main__":
    for p in write_static_fixtures():
        print("wrote", p)
