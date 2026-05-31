# Stage 4 — Cekura evaluation

## What's built and passing now (local, deterministic)

The reasoning/honesty/voice quality of Vantage is proven by a **local deterministic
harness** that hits the real Nemotron-3-Super LLM with the bot's real system prompt
and the real `room_query` tools, pointed at fixed scenes:

```bash
cd /Applications/vantage/yc-voice-agents-hackathon/server
uv run python ../../vantage/evals/run_evals.py
```

It activates a fixture per scenario (`VANTAGE_ROOM_STATE`), runs the tool-calling
loop, and scores **correctness + honesty (LLM judge with ground truth) + voice
(rule-based)**. Exits non-zero if anything fails, so it's an iterate-until-green loop.

Fixtures live in `vantage/fixtures/` (`fixture_desk.json`, `fixture_empty.json`,
`fixture_changed.json`). To point the *live bot* at one for a manual repeatable run:

```bash
cd /Applications/vantage && uv run python -m vantage.fixtures.activate desk
export VANTAGE_ROOM_STATE=/Applications/vantage/vantage/fixtures/active.json
# then start the bot (no vision loop needed — it reads the fixture):
cd yc-voice-agents-hackathon/server && uv run python ../bot_roomsense.py
```

## Running the *actual* Cekura cloud evals — what it needs from you

Two things block driving Cekura from here; both are on your side:

1. **The Cekura skills/MCP aren't installed in this Claude Code session.** Install the
   plugin (it bundles the skills + auto-configures the MCP):
   ```
   /plugin marketplace add cekura-ai/cekura-skills
   /plugin install cekura@cekura-skills
   ```
   Then `/cekura-report` drives agent creation + test runs.

2. **Cekura connects to Pipecat agents through Pipecat Cloud, not localhost.** Per
   docs.cekura.ai, you configure provider = **Pipecat**, a **Pipecat Cloud API key**,
   and a **Pipecat Agent Name** — Cekura spins up sessions via Daily/Pipecat Cloud.
   That conflicts with the "local only / ignore Pipecat Cloud" rule. To use Cekura
   cloud you'd deploy `bot_roomsense.py` to Pipecat Cloud first
   (`pc cloud deploy`), with `VANTAGE_ROOM_STATE` baked to a fixture for repeatable
   scoring. If you want to stay local, the deterministic harness above is the
   substitute and tests the same reasoning/honesty.

## The evalset

`vantage/evals/cekura_evalset.json` is the scenario + metric spec (same 4 scenarios,
3 metrics) to recreate in Cekura once connected. Each scenario names the fixture to
pin via `VANTAGE_ROOM_STATE` so Cekura's runs are deterministic too.
