"""Deterministic room_state fixtures for reproducible evaluation.

The agent's answers normally depend on the live camera, which an eval harness
can't control. These fixtures stand in for known scenes so Vantage's reasoning
and honesty can be tested repeatably. Point the bot at one with:

    export VANTAGE_ROOM_STATE=/Applications/vantage/vantage/fixtures/active.json
    python -m vantage.fixtures.activate desk      # stamps fresh timestamps
"""
