"""Activate a fixture scene for reproducible runs.

Writes ``active.json`` (a fresh-timestamped room_state) that you point the bot at:

    python -m vantage.fixtures.activate desk
    export VANTAGE_ROOM_STATE=/Applications/vantage/vantage/fixtures/active.json

For the ``changed`` scene it also seeds ``change_snapshot.json`` with the DESK
baseline, so a single what_changed() call deterministically reports the charger
as missing and the phone as new (simulating "since earlier the desk had a
charger").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vantage.fixtures.scenes import FIXTURE_DIR, baseline_snapshot, build_state

ACTIVE_PATH = FIXTURE_DIR / "active.json"
SNAPSHOT_PATH = FIXTURE_DIR / "change_snapshot.json"


def activate(name: str, now: float | None = None) -> Path:
    now = time.time() if now is None else now
    ACTIVE_PATH.write_text(json.dumps(build_state(name, now), indent=2))

    # The "changed" scene only makes sense relative to the desk baseline.
    if name == "changed":
        SNAPSHOT_PATH.write_text(json.dumps(baseline_snapshot("desk", now - 120.0)))
    elif SNAPSHOT_PATH.exists():
        # Clear any stale baseline so other scenes start fresh.
        SNAPSHOT_PATH.unlink()
    return ACTIVE_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Activate a Vantage fixture scene.")
    ap.add_argument("scene", choices=["desk", "empty", "changed"])
    args = ap.parse_args()
    path = activate(args.scene)
    print(f"Activated '{args.scene}' -> {path}")
    print(f"export VANTAGE_ROOM_STATE={path}")
