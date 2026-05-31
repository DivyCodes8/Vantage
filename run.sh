#!/bin/bash
# Vantage — runs vision loop + voice bot together. Ctrl-C stops both.
source "$HOME/.local/bin/env" 2>/dev/null
trap "kill 0" EXIT

( cd /Applications/vantage && uv run python -m vantage.vision.vision_loop ) &
( cd /Applications/vantage/yc-voice-agents-hackathon/server && uv run python ../bot_roomsense.py ) &

wait
