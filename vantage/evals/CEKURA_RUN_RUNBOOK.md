# Cekura run runbook — paste-and-go for the NEW session

Context for the fresh Claude Code session that will run the Cekura evals (this is
separate from the session building the Vantage frontend). Everything is already
deployed and installed; you just need to drive Cekura.

## Preconditions (already done)
- Pipecat Cloud agent **`vantage-eval`** is LIVE (org `awake-swordfish-tomato-814`,
  us-west). It reads a FIXTURE, not a camera, and switches scene per session via a
  body `{"fixture":"desk|empty|changed"}`.
- Cekura plugin installed + enabled; Cekura MCP registered (OAuth).
- Pipecat Cloud **public API key** for Cekura: read it from
  `/Applications/vantage/deploy/pipecat_public_key.txt` (a `pk_...` value).

## Step 0 — authenticate Cekura MCP (one-time, this session)
Run `/mcp` → **cekura** → **Authenticate** → sign in at dashboard.cekura.ai →
**Authorize**. Confirm with `mcp__cekura__list_available_tools` (should return tools).

## Step 1 — connect the Pipecat agent in Cekura
Use the plugin: run `/cekura-onboarding` (or `/cekura-report`, which also onboards).
When it asks for the agent connection, provide:
- **Provider:** Pipecat
- **Pipecat Cloud API key:** the `pk_...` from `deploy/pipecat_public_key.txt`
- **Pipecat Agent Name:** `vantage-eval`

## Step 2 — create the 4 evaluators/scenarios
Source spec: `/Applications/vantage/vantage/evals/cekura_evalset.json`. Create one
scenario per entry; for EACH, set the **Agent Configuration JSON** to the fixture:

| Utterance | Agent-Config body | Must |
|---|---|---|
| Where's my charger? | `{"fixture":"desk"}` | locate charger on the right |
| Where's my charger? | `{"fixture":"empty"}` | admit it hasn't seen it |
| Is anything unsafe? | `{"fixture":"desk"}` | flag bottle-near-laptop spill |
| What changed since earlier? | `{"fixture":"changed"}` | charger gone + phone new |

Metrics: correctness, honesty (no invented objects / admits absence), voice (1–2
short spoken sentences, no raw numbers). See cekura_evalset.json for details.
> If Cekura only allows ONE global agent-config, run scenarios in groups by fixture
> (change config between groups), or deploy fixture-specific agents.

## Step 3 — run + read
`/run-evals` (or `/cekura-report`). Read transcripts + scores.

## Step 4 — iterate until green (only if something fails)
Edit the prompt at `/Applications/vantage/vantage/agent_prompt.py` (single source —
the local bot uses it too). Then redeploy the cloud copy:
```bash
bash /Applications/vantage/deploy/build.sh
cd /Applications/vantage/deploy
script -q /dev/null pc cloud deploy --secrets vantage-eval-secrets -y   # pty needed; pc crashes without a TTY
```
Re-run the Cekura evals. NOTE: the LOCAL harness predicts results without a
redeploy — `cd /Applications/vantage/yc-voice-agents-hackathon/server && uv run
python ../../vantage/evals/run_evals.py` (already 4/4). The cloud copy shares the
exact prompt/tools/fixtures, so local green ≈ cloud green.

## Gotchas
- `pc` CLI needs a TTY: wrap every `pc ...` in `script -q /dev/null pc ...` and use
  non-interactive flags (`--skip`, `-y`).
- Don't touch `server/`, `run.sh`, the live bot, or the vision module.
