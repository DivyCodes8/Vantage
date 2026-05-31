# Vantage: Spatial Voice Assistant

Vantage turns any camera into a pair of eyes you can talk to. It watches a space in real time, builds a memory of what is in it and where, and answers spoken questions out loud. You can ask what is around you, where you left something, what has changed, or whether anything looks unsafe, and it replies in a sentence or two. No screen, no scrolling, no searching. You just ask.

For someone who cannot see the room, or cannot stop to look, that difference is the whole point. Vantage is built so that understanding your surroundings does not depend on sight or a free pair of hands.

**Ask it things like:**
- *"What do you see?"*
- *"Where's my charger?"*
- *"What changed since earlier?"*
- *"Is anything unsafe?"*

It answers out loud in natural speech, grounded only in what the camera has actually seen. No screen needed, and it does not guess.

---

## Who it's for

- **People who are blind or have low vision.** Instead of feeling around a room, ask what is in front of you and where it is. Vantage describes the scene, locates specific objects on the left, center, or right, and tells you whether something it saw earlier is still there.
- **People who can't easily look right now.** Hands full, cooking, carrying something, recovering from an injury, or moving with limited mobility. You can check on a space without stopping what you are doing.
- **Older adults living independently, and the people who care for them.** A quick "is anything unsafe?" can catch a spill or trip hazard before it causes a fall, and a carer can check on a room without hovering.
- **Anyone who wants a second set of eyes.** A bottle of water next to a laptop, a cable across the floor, an object balanced near an edge. Vantage flags these before they turn into a spill, a trip, or a fall.

The common thread is simple. When looking is hard, slow, or unsafe, Vantage lets you ask instead.

---

## What makes it interesting

Most camera assistants describe a single frame and forget it. **Vantage remembers.** It keeps a running, time-stamped picture of the room, so it can answer questions about the past as well as the present. "Where's my charger?" still works when the charger is out of view right now, because Vantage knows where it last saw it and how long ago.

The harder problem was **honesty**. A voice assistant that confidently invents a location is worse than useless for someone who depends on it, and dangerous for someone who cannot check by looking. Vantage is built to never claim something it did not see. Every answer is grounded in structured detection data that carries confidence and visibility flags, and the language model follows strict rules. If confidence is low it hedges and suggests moving the camera closer. If an object has not been seen it says so plainly. It never fills a gap with a guess.

---

## How it works

Vantage runs two things side by side:

1. **Vision loop.** Watches the webcam, detects objects with YOLOE-26 (open-vocabulary detector), generates a scene caption with Florence-2, and writes everything to a `room_state.json` file.
2. **Voice agent.** Listens to your spoken question via Nemotron ASR, reasons about the room using the latest `room_state.json`, and speaks the answer back via Gradium TTS. It never invents objects. It only answers from what the camera actually saw.

The two processes are fully independent. The voice agent reads a plain JSON file, and no vision models are loaded into the voice pipeline.

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
| Evaluation | Cekura |

---

## How to run it

**Requirements:** Mac with a webcam. Python 3.11+ and `uv` handle the rest.

```bash
# 1. Clone
git clone https://github.com/DivyCodes8/Vantage.git
cd Vantage

# 2. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Install dependencies (two separate envs, vision and voice stay isolated)
cd yc-voice-agents-hackathon/server && uv sync && cd ../..
uv sync

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
| "What do you see?" | Description of currently visible objects plus scene caption |
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

**Result: 4/4 scenarios passing.** Charger location, empty-room honesty, spill hazard detection, and change detection. Each scored on correctness, honesty (never invents objects), and voice quality (one or two short spoken sentences, no raw numbers).

The eval copy was also deployed to Pipecat Cloud and tested live via Cekura, with the same results.

---

## Feedback

Built for the **YC Voice Agents Hackathon** (May 2026), hosted by Cekura and Daily, in partnership with NVIDIA, AWS, and Twilio.

### Tools used

- **Pipecat:** voice pipeline orchestration, SmallWebRTC transport, Gradium TTS integration.
- **NVIDIA Nemotron:** speech-to-text (Nemotron Speech Streaming) and LLM (Nemotron-3-Super-120B), both via the provided hackathon endpoints.
- **Gradium:** text-to-speech with natural, low-latency voice output.
- **Cekura:** automated voice agent evaluation across 4 scenarios with correctness, honesty, and voice metrics.
- **Ultralytics YOLOE-26:** open-vocabulary object detection, prompt-free mode, Apple Silicon MPS.
- **Microsoft Florence-2:** scene captioning for ambient context.

### What we learned / feedback

The hardest part wasn't the voice pipeline. Pipecat made that straightforward. The real challenge was **honesty at the boundary**: making the agent admit it hasn't seen something rather than hallucinate a location. The combination of typed tool outputs (`low_confidence`, `visible: false`, `found: false`) and explicit grounding rules in the system prompt solved it cleanly.

YOLOE-26's prompt-free vocabulary is noisy in a real room (it emits abstract labels alongside real objects). The query layer's fuzzy matching and synonym expansion handles this well, but it's worth noting for anyone building on top of this approach.

Cekura's evaluation framework was the right forcing function for getting the honesty guarantees actually testable and repeatable. The deterministic fixture approach (swap `VANTAGE_ROOM_STATE` to a known scene) is reusable for any vision-dependent voice agent.
