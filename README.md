# Vantage — Spatial Voice Assistant

Vantage is an app that uses a camera to understand a room and answer spoken questions about it, so people can stay aware of their surroundings hands-free.

**Ask it things like:**
- *"What do you see?"*
- *"Where's my charger?"*
- *"What changed since earlier?"*
- *"Is anything unsafe?"*

It answers out loud in natural speech — no screen needed.

---

## Who it's for

- **People who are blind or have low vision** — get a spoken description of what's in the room and where things are.
- **People who can't easily look** — hands full, limited mobility, or just busy — ask without stopping what you're doing.
- **Anyone who wants a quick safety check** — spill risks near electronics, trip hazards, objects near edges — Vantage flags them before they become problems.

---

## How it works

Vantage runs two things side by side:

1. **Vision loop** — watches the webcam, detects objects with YOLOE-26 (open-vocabulary detector), generates a scene caption with Florence-2, and writes everything to a `room_state.json` file.
2. **Voice agent** — listens to your spoken question via Nemotron ASR, reasons about the room using the latest `room_state.json`, and speaks the answer back via Gradium TTS. It never invents objects — it only answers from what the camera actually saw.

The two processes are fully independent. The voice agent reads a plain JSON file; no vision models are loaded into the voice pipeline.

---

## Tech stack

| Layer | Tech |
|---|---|
| Speech-to-text | NVIDIA Nemotron Speech Streaming (free, no key) |
| LLM | NVIDIA Nemotron-3-Super-120B (free, no key) |
| Text-to-speech | Gradium (native Pipecat integration) |
| Object detection | YOLOE-26 via Ultralytics (local, Apple Silicon MPS) |
| Scene captioning | Microsoft Florence-2-base (local, MPS) |
| Voice pipeline | Pipecat |
| Transport | SmallWebRTC (local) |

---

## How to run it

**Requirements:** Mac with a webcam. Python 3.11+ and `uv` are installed automatically.

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/vantage.git
cd vantage

# 2. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Install dependencies (two separate envs — vision and voice stay isolated)
cd yc-voice-agents-hackathon/server && uv sync && cd ../..
cd /path/to/vantage && uv sync && cd -

# 4. Add your Gradium API key
# Edit yc-voice-agents-hackathon/server/.env and set:
#   GRADIUM_API_KEY=your_key_here
# Get a free key at https://gradium.ai

# 5. Run everything with one command
bash run.sh
```

Then open **http://localhost:7860** in your browser and click **Connect**. The camera window opens automatically in a separate window.

Point the camera at your room and start talking.

---

## Project structure

```
vantage/
├── run.sh                          # one-command launcher (Ctrl-C stops both)
├── vantage/
│   ├── vision/
│   │   ├── vision_loop.py          # webcam → YOLOE detection + Florence caption
│   │   ├── detector.py             # YOLOE-26 wrapper (prompt-free + set_prompts)
│   │   └── scene.py                # Florence-2-base scene captioner
│   ├── memory/
│   │   ├── room_state.py           # atomic JSON store for detected objects
│   │   └── room_query.py           # query layer (fuzzy match, hazard heuristics)
│   ├── agent_prompt.py             # single-source system prompt
│   ├── fixtures/                   # deterministic test scenes
│   └── evals/                      # local eval harness (4/4 passing)
├── yc-voice-agents-hackathon/
│   ├── server/                     # provided Pipecat starter (untouched)
│   └── bot_roomsense.py            # Vantage voice bot (reads room_state.json)
└── deploy/                         # Pipecat Cloud eval copy (no vision deps)
```

---

## Demo questions to try

| Ask this | What to expect |
|---|---|
| "What do you see?" | Description of currently visible objects + scene caption |
| "Where's my [object]?" | Location (left / center / right) and whether it's visible now or was seen recently |
| "Is anything unsafe?" | Flags spill risks, trip hazards, and items near edges |
| "What changed since earlier?" | New objects, missing objects, and anything that moved |

---

## Evaluation

Vantage was tested with a deterministic eval harness using fixture room states (no live camera needed):

```bash
cd yc-voice-agents-hackathon/server
uv run python ../../vantage/evals/run_evals.py
```

**Result: 4/4 scenarios passing** — charger location, empty-room honesty, spill hazard detection, and change detection. Each scored on correctness, honesty (never invents objects), and voice quality (1–2 short spoken sentences, no raw numbers).

The eval copy was also deployed to Pipecat Cloud and tested live via Cekura, with the same results.

---

## Hackathon context

Built for the **YC Voice Agents Hackathon** (May 2026), hosted by Cekura and Daily, in partnership with NVIDIA, AWS, and Twilio.

### Tools used

- **Pipecat** — voice pipeline orchestration, SmallWebRTC transport, Gradium TTS integration
- **NVIDIA Nemotron** — speech-to-text (Nemotron Speech Streaming) and LLM (Nemotron-3-Super-120B), both via the provided hackathon endpoints
- **Gradium** — text-to-speech with natural, low-latency voice output
- **Cekura** — automated voice agent evaluation; 4 scenarios with correctness, honesty, and voice metrics
- **Ultralytics YOLOE-26** — open-vocabulary object detection, prompt-free mode, Apple Silicon MPS
- **Microsoft Florence-2** — scene captioning for ambient context

### What we learned / feedback

The hardest part wasn't the voice pipeline — Pipecat made that straightforward. The real challenge was **honesty at the boundary**: making the agent admit it hasn't seen something rather than hallucinate a location. The combination of typed tool outputs (`low_confidence`, `visible: false`, `found: false`) and explicit grounding rules in the system prompt solved it cleanly.

YOLOE-26's prompt-free vocabulary is noisy in a real room (it emits abstract labels alongside real objects). The query layer's fuzzy matching and synonym expansion handles this well, but it's worth noting for anyone building on top of this approach.

Cekura's evaluation framework was the right forcing function for getting the honesty guarantees actually testable and repeatable. The deterministic fixture approach (swap `VANTAGE_ROOM_STATE` to a known scene) is reusable for any vision-dependent voice agent.
