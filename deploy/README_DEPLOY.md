# Vantage eval — Pipecat Cloud deploy + Cekura

A **self-contained, torch-free** copy of the local room-sense bot. Logic-identical
to `bot_roomsense.py` (same prompt via `vantage/agent_prompt.py`, same tools, same
fixtures); the ONLY differences are: it reads a **fixture** instead of the live
camera, and it accepts **Daily** WebRTC sessions (how Pipecat Cloud / Cekura
connect). Building/deploying this never touches `server/`, `run.sh`, the live bot,
or the vision module.

One deployed agent serves all scenarios: Cekura passes `{"fixture":"desk"|"empty"|
"changed"}` per scenario in the **Agent Configuration JSON** (→ the session `body`),
and the bot switches scenes (and seeds the change-baseline for `changed`).

## A. Keep the bundle in sync (run after any prompt change)
```bash
bash /Applications/vantage/deploy/build.sh   # copies latest agent_prompt + room_query + fixtures
```

## B. Deploy to Pipecat Cloud
```bash
# 1. Install the Pipecat CLI (one-time)
uv tool install pipecat-ai-cli

# 2. Log in  ← STOP: this opens a browser for YOUR Pipecat Cloud account
pc cloud auth login

# 3. Upload secrets  ← STOP: paste your Gradium key first
cd /Applications/vantage/deploy
cp secrets.env.example secrets.env      # then edit secrets.env, set GRADIUM_API_KEY=...
pc cloud secrets set vantage-eval-secrets --file secrets.env

# 4. Deploy (Pipecat builds the image in the cloud)
pc cloud deploy

# 5. Verify it's live
pc cloud agent list                     # expect: vantage-eval
pc cloud agent status vantage-eval      # or check https://pipecat.daily.co
```
The agent name is **`vantage-eval`** (from `pcc-deploy.toml`).

## C. Connect Cekura
1. Install the plugin in Claude Code:
   ```
   /plugin marketplace add cekura-ai/cekura-skills
   /plugin install cekura@cekura-skills
   ```
2. In Cekura, add the agent with **provider = Pipecat**, your **Pipecat Cloud API
   key**, and **Pipecat Agent Name = `vantage-eval`**.
3. Recreate the 4 scenarios from `vantage/evals/cekura_evalset.json`. For each,
   set the **Agent Configuration JSON** to the matching fixture:
   - `Where's my charger?` → `{"fixture":"desk"}` (must locate it)
   - `Where's my charger?` → `{"fixture":"empty"}` (must admit not seen)
   - `Is anything unsafe?` → `{"fixture":"desk"}` (must flag bottle-near-laptop spill)
   - `What changed since earlier?` → `{"fixture":"changed"}` (charger gone + phone new)
   > If Cekura only allows ONE global agent-config, run the scenarios in groups by
   > fixture (change the config between groups), or deploy fixture-specific agents.
4. Run the simulations and read the report.

## D. Iterate
Edit the prompt in `/Applications/vantage/vantage/agent_prompt.py` (single source —
the local bot uses it too), then `bash build.sh && pc cloud deploy`, and re-run
Cekura. The local harness (`vantage/evals/run_evals.py`) predicts results without a
redeploy, since the cloud copy shares the exact prompt/tools/fixtures.
