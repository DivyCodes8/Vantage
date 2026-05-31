"""Read-only query layer over room_state.json for the voice agent.

PURE STANDARD LIBRARY — safe to import from the bot's (pipecat) venv without
pulling in torch/ultralytics. It only ever READS room_state.json (written by the
vision loop); it never modifies the vision module. The only file it writes is its
own ``change_snapshot.json`` used by :func:`what_changed`.

Every public function loads room_state.json FRESH on each call, so answers always
reflect the latest frame, and degrades gracefully if the file is missing, empty,
or caught mid-write (atomic rename in RoomState makes the last case rare).

The three functions return compact, LLM-friendly dicts. They deliberately do NOT
produce prose — the bot's system prompt turns this data into one or two spoken
sentences.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

# Reuse the canonical room_state.json location and threshold from the store.
# (room_state.py is pure stdlib, so this import is torch-free.)
from vantage.memory.room_state import CONFIDENCE_THRESHOLD, DEFAULT_PATH

def _state_path() -> Path:
    """Resolve the room_state.json to read, fresh each call.

    VANTAGE_ROOM_STATE lets the bot (or the eval harness) point at a fixture for
    reproducible runs; otherwise the live vision loop's default file is used.
    """
    return Path(os.environ.get("VANTAGE_ROOM_STATE") or DEFAULT_PATH)


def _snapshot_path() -> Path:
    """where what_changed() stores its baseline (next to the active state file)."""
    override = os.environ.get("VANTAGE_SNAPSHOT")
    return Path(override) if override else _state_path().parent / "change_snapshot.json"

# A small synonym map so a spoken word maps to whatever label the detector used.
# Keys and values are all lowercase. Matching is bidirectional within a group.
SYNONYMS: dict[str, list[str]] = {
    "charger": ["cable", "power adapter", "adapter", "power cord",
                "charging cable", "usb", "charger cable"],
    "phone": ["cellphone", "cell phone", "smartphone", "mobile", "iphone", "beeper", "gadget"],
    "cup": ["mug", "glass", "coffee", "tea", "tumbler"],
    "bottle": ["water bottle", "flask", "thermos", "drink"],
    "laptop": ["computer", "notebook", "macbook", "pc"],
    "keys": ["keychain", "key", "car key"],
    "remote": ["controller", "remote control", "clicker"],
    "glasses": ["spectacles", "eyeglasses", "sunglasses", "eyewear"],
    "wallet": ["purse", "billfold"],
    "pen": ["pencil", "marker", "biro"],
    "headphones": ["earphones", "earbuds", "headset", "earplug", "airpods"],
    "lamp": ["light", "desk lamp"],
    "camera": ["webcam", "cam"],
    "book": ["notebook", "textbook"],
    "bag": ["backpack", "handbag", "rucksack"],
}

# Hazard heuristic vocabularies (substring matched against labels).
LIQUID_WORDS = ["cup", "mug", "glass", "bottle", "coffee", "tea", "water", "drink",
                "can", "soda", "beer", "wine", "jar"]
ELECTRONIC_WORDS = ["laptop", "computer", "keyboard", "phone", "monitor", "tablet",
                    "charger", "cable", "tv", "television", "mouse", "console",
                    "camera", "gadget", "speaker", "router"]
CABLE_WORDS = ["cable", "cord", "wire", "rope", "extension cord", "charging cable", "lead"]
SURFACE_WORDS = ["table", "desk", "counter", "countertop", "shelf", "stand", "nightstand"]


# --------------------------------------------------------------------- loading
def _load_state() -> dict | None:
    """Read room_state.json fresh. Returns None if missing/empty/corrupt."""
    try:
        path = _state_path()
        if not path.exists():
            return None
        text = path.read_text()
        if not text.strip():
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        # Missing, unreadable, or caught mid-write — caller treats as "blind".
        return None


def _visible(objs: list[dict]) -> list[dict]:
    return [o for o in objs if o.get("visible")]


def _compact(o: dict, now: float) -> dict:
    return {
        "label": o.get("label"),
        "region": o.get("region"),
        "last_seen_seconds_ago": max(0, int(now - float(o.get("last_seen", now)))),
        "confidence": round(float(o.get("confidence", 0.0)), 2),
        "low_confidence": bool(o.get("low_confidence", False)),
        "visible": bool(o.get("visible", False)),
    }


# ------------------------------------------------------------------- matching
def _expand_terms(query: str) -> set[str]:
    """Turn a spoken query into a set of candidate match terms via synonyms."""
    q = (query or "").lower().strip()
    terms: set[str] = set()
    # raw query + its individual words
    if q:
        terms.add(q)
        terms.update(w for w in q.replace("?", " ").replace(",", " ").split() if len(w) >= 3)
    # pull in whole synonym groups that the query touches
    for key, syns in SYNONYMS.items():
        group = {key, *syns}
        if any(g in q for g in group) or any(q and q in g for g in group):
            terms |= group
    # drop tiny stop-ish tokens
    return {t for t in terms if len(t) >= 3}


_GENERAL_MARKERS = [
    "what do you see", "what can you see", "what's around", "whats around",
    "around me", "what is around", "everything", "describe", "the room",
    "what's in", "whats in", "what is in", "look around", "see right now",
    "in front of me", "on my desk", "whats here", "what's here",
]


def _is_general(query: str) -> bool:
    q = (query or "").lower().strip()
    if q in ("", "scene", "room", "desk"):
        return True
    return any(m in q for m in _GENERAL_MARKERS)


def _matches(label: str, terms: set[str]) -> bool:
    """Match a label against candidate terms.

    Short terms (<5 chars) must match at a word boundary so "cup" doesn't hit
    "cupboard" and "plug"-like fragments don't hit unrelated labels. Longer
    terms may match as a substring, which is safe enough in practice.
    """
    L = (label or "").lower()
    if not L:
        return False
    label_words = set(L.replace("-", " ").replace("/", " ").split())
    for t in terms:
        if t == L:
            return True
        if " " in t and t in L:  # multiword phrase contained in label
            return True
        if len(t) >= 5 and (t in L or L in t):
            return True
        # word-level match for short tokens
        term_words = set(t.split())
        if term_words & label_words:
            return True
    return False


# -------------------------------------------------------------- public tools
def query_room_state(query: str) -> dict:
    """Find an object, or summarize the scene. See module docstring for shape."""
    data = _load_state()
    now = time.time()
    if data is None:
        return {"found": False, "blind": True,
                "message": "No room data yet — I can't see anything."}

    objs = data.get("objects", [])
    caption = data.get("scene_caption", "")

    # General "what do you see / what's around me" → currently visible objects.
    if _is_general(query):
        visible = [_compact(o, now) for o in _visible(objs)]
        return {
            "found": bool(visible) or bool(caption),
            "mode": "scene",
            "scene_caption": caption,
            "visible_objects": visible,
        }

    # Specific object lookup (fuzzy via synonyms + substring).
    terms = _expand_terms(query)
    matched = [o for o in objs if _matches(o.get("label", ""), terms)]
    if not matched:
        return {"found": False, "query": query, "scene_caption": caption}

    # Prefer visible + higher confidence.
    matched.sort(key=lambda o: (o.get("visible", False), o.get("confidence", 0.0)), reverse=True)
    return {
        "found": True,
        "mode": "object",
        "query": query,
        "scene_caption": caption,
        "matches": [_compact(o, now) for o in matched[:5]],
    }


def what_changed() -> dict:
    """Diff currently-visible objects against the previous snapshot.

    Returns new / missing / moved object lists and stores the current frame as
    the next baseline. The first call just establishes a baseline.
    """
    data = _load_state()
    now = time.time()
    if data is None:
        return {"available": False, "message": "No room data yet — I can't see anything."}

    current = {o["label"]: o for o in _visible(data.get("objects", [])) if o.get("label")}
    snap = _load_snapshot()
    # Persist the new baseline AFTER we've read the old one.
    _save_snapshot({
        "taken": now,
        "objects": {lbl: {"region": o.get("region")} for lbl, o in current.items()},
    })

    if snap is None:
        return {"available": True, "baseline": True, "new": [], "missing": [], "moved": [],
                "message": "Just started watching — nothing to compare against yet."}

    prev = snap.get("objects", {})
    new = [{"label": l, "region": current[l].get("region")} for l in current if l not in prev]
    missing = [{"label": l, "region": prev[l].get("region")} for l in prev if l not in current]
    moved = [
        {"label": l, "from": prev[l].get("region"), "to": current[l].get("region")}
        for l in current
        if l in prev and current[l].get("region") != prev[l].get("region")
    ]
    return {
        "available": True,
        "baseline": False,
        "seconds_since_snapshot": max(0, int(now - float(snap.get("taken", now)))),
        "new": new,
        "missing": missing,
        "moved": moved,
    }


def check_hazards() -> dict:
    """Surface candidate safety risks from spatial heuristics.

    Provides raw flags (spill / trip / fall risk) for the LLM to reason about and
    rank — it does NOT make the final call. Operates on currently-visible objects.
    """
    data = _load_state()
    now = time.time()
    if data is None:
        return {"available": False, "message": "No room data yet — I can't see anything."}

    objs = _visible(data.get("objects", []))
    caption = data.get("scene_caption", "")
    # Estimate frame extent from the widest/tallest bbox seen (no frame size is
    # stored in room_state, and we may not modify the vision module).
    fw = max((float(o["bbox"][2]) for o in objs if o.get("bbox")), default=1920.0)
    fh = max((float(o["bbox"][3]) for o in objs if o.get("bbox")), default=1080.0)
    frame_area = max(fw * fh, 1.0)

    # Ignore full-frame "scene" blobs (e.g. "darkness", "ceiling") as hazards.
    cand = [o for o in objs if o.get("bbox") and _area(o["bbox"]) < 0.8 * frame_area]

    hazards: list[dict] = []

    liquids = [o for o in cand if _in_set(o["label"], LIQUID_WORDS)]
    electronics = [o for o in cand if _in_set(o["label"], ELECTRONIC_WORDS)]
    for liq in liquids:
        for elec in electronics:
            if _near(liq["bbox"], elec["bbox"], fw):
                hazards.append({
                    "type": "spill_risk",
                    "items": [liq["label"], elec["label"]],
                    "where": liq.get("region"),
                    "note": f'{liq["label"]} is close to {elec["label"]}',
                })

    for o in cand:
        b = o["bbox"]
        if _in_set(o["label"], CABLE_WORDS) and _width(b) > 0.4 * fw and b[3] > 0.5 * fh:
            hazards.append({
                "type": "trip_hazard",
                "items": [o["label"]],
                "where": o.get("region"),
                "note": f'{o["label"]} stretches across the lower floor area',
            })

    # Fall risk: prefer table-relative; an object near a detected surface's left
    # or right edge. If no surface is detected, fall back to objects hugging the
    # left/right frame border.
    surfaces = [o for o in cand if _in_set(o["label"], SURFACE_WORDS)]
    for o in cand:
        if _in_set(o["label"], SURFACE_WORDS):
            continue
        b = o["bbox"]
        flagged = False
        for s in surfaces:
            sb = s["bbox"]
            if _overlaps_x(b, sb) and _near_side_edge_of(b, sb):
                flagged = True
                break
        if not flagged and not surfaces:
            margin = 0.03 * fw
            if b[0] < margin or b[2] > fw - margin:
                flagged = True
        if flagged:
            hazards.append({
                "type": "fall_risk",
                "items": [o["label"]],
                "where": o.get("region"),
                "note": f'{o["label"]} sits near an edge and could fall',
            })

    return {
        "available": True,
        "scene_caption": caption,
        "objects": [_compact(o, now) for o in cand],
        "hazards": hazards,
    }


# ------------------------------------------------------------------ geometry
def _area(b) -> float:
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))


def _width(b) -> float:
    return b[2] - b[0]


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _near(a, b, fw, frac=0.22) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by) < frac * fw


def _overlaps_x(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0])


def _near_side_edge_of(obj, surf, frac=0.12) -> bool:
    span = max(surf[2] - surf[0], 1.0)
    ox, _ = _center(obj)
    return (ox - surf[0]) < frac * span or (surf[2] - ox) < frac * span


def _in_set(label: str, words) -> bool:
    L = (label or "").lower()
    return any(w in L for w in words)


# ------------------------------------------------------------------ snapshot
def _load_snapshot() -> dict | None:
    try:
        path = _snapshot_path()
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _save_snapshot(payload: dict) -> None:
    path = _snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".snap.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "what do you see"
    print("query_room_state(%r):" % q, json.dumps(query_room_state(q), indent=2))
    print("\ncheck_hazards():", json.dumps(check_hazards(), indent=2))
    print("\nwhat_changed():", json.dumps(what_changed(), indent=2))
