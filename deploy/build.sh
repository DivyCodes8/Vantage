#!/bin/bash
# Sync the latest local sources into this self-contained deploy bundle, so the
# cloud copy stays logic-identical to the local bot. Run this before deploying
# (and any time you iterate the prompt in vantage/agent_prompt.py).
#
# It NEVER modifies server/ or the local bot — it only copies OUT of them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC_SERVER="/Applications/vantage/yc-voice-agents-hackathon/server"
SRC_VANTAGE="/Applications/vantage/vantage"

# Vendored provided services (verbatim copies; server/ is untouched).
cp "$SRC_SERVER/nemotron_llm.py" "$HERE/nemotron_llm.py"
cp "$SRC_SERVER/nvidia_stt.py"   "$HERE/nvidia_stt.py"

# Vantage stdlib subset only — NO vision modules (detector/scene/vision_loop).
mkdir -p "$HERE/vantage/memory" "$HERE/vantage/fixtures"
cp "$SRC_VANTAGE/__init__.py"             "$HERE/vantage/__init__.py"
cp "$SRC_VANTAGE/agent_prompt.py"         "$HERE/vantage/agent_prompt.py"
cp "$SRC_VANTAGE/memory/__init__.py"      "$HERE/vantage/memory/__init__.py"
cp "$SRC_VANTAGE/memory/room_query.py"    "$HERE/vantage/memory/room_query.py"
cp "$SRC_VANTAGE/memory/room_state.py"    "$HERE/vantage/memory/room_state.py"
cp "$SRC_VANTAGE/fixtures/__init__.py"    "$HERE/vantage/fixtures/__init__.py"
cp "$SRC_VANTAGE/fixtures/fixture_desk.json"    "$HERE/vantage/fixtures/"
cp "$SRC_VANTAGE/fixtures/fixture_empty.json"   "$HERE/vantage/fixtures/"
cp "$SRC_VANTAGE/fixtures/fixture_changed.json" "$HERE/vantage/fixtures/"

echo "Bundle synced into $HERE"
