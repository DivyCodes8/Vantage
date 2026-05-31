"""Deterministic eval harness for the Vantage room-sense agent.

Mirrors a Cekura evalset, but runs locally and reproducibly: for each scenario we
point room_query at a fixture, run the REAL Nemotron-3-Super LLM with the bot's
ACTUAL system prompt + tools, execute the REAL room_query tools, and score the
spoken answer on three metrics:

  * correctness  — did it convey the right answer for this scene? (LLM judge w/ ground truth)
  * honesty      — did it avoid inventing objects / admit absence? (LLM judge w/ ground truth)
  * voice        — 1-2 short spoken sentences, no markdown/lists, no raw numbers (rule-based)

Run it from the server venv (which has openai + pipecat):
    cd yc-voice-agents-hackathon/server
    uv run python ../../vantage/evals/run_evals.py

Exits non-zero if any scenario fails (so you can iterate until green).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_VANTAGE_ROOT = _HERE.parents[2]                     # /Applications/vantage
_REPO = _VANTAGE_ROOT / "yc-voice-agents-hackathon"
_SERVER = _REPO / "server"
for _p in (str(_VANTAGE_ROOT), str(_REPO), str(_SERVER)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os

from dotenv import load_dotenv

load_dotenv(_SERVER / ".env", override=True)

from openai import OpenAI

import vantage.memory.room_query as rq
from vantage.fixtures.activate import ACTIVE_PATH, SNAPSHOT_PATH, activate

# Import the bot's real system prompt (source of truth — iterate by editing the bot).
import bot_roomsense  # noqa: E402

SYSTEM = bot_roomsense.SYSTEM_INSTRUCTION

MODEL = os.getenv("NEMOTRON_LLM_MODEL", "nvidia/nemotron-3-super")
client = OpenAI(api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),
                base_url=os.environ["NEMOTRON_LLM_URL"])
_EXTRA = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

# --- Tool schemas the agent sees (mirror bot_roomsense's registered tools) -----
TOOLS = [
    {"type": "function", "function": {
        "name": "query_room_state",
        "description": "Look up what the camera currently sees, or find a specific object "
                       "in the room. Use for 'what do you see', 'what's around me', and "
                       "'where's my <object>'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Object name (e.g. 'charger') or a "
                                                       "general phrase like 'what do you see'."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "what_changed",
        "description": "Report what newly appeared, went missing, or moved since you last "
                       "checked. Use for 'what changed?'.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_hazards",
        "description": "Check for possible spill/trip/fall risks from object positions. Use "
                       "for 'is anything unsafe?'. Returns candidate risks to rank.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _exec_tool(name: str, args: dict) -> dict:
    if name == "query_room_state":
        return rq.query_room_state(args.get("query", ""))
    if name == "what_changed":
        return rq.what_changed()
    if name == "check_hazards":
        return rq.check_hazards()
    return {"error": f"unknown tool {name}"}


def run_agent(question: str, max_steps: int = 4):
    """Run the tool-calling loop; return (final_answer, tool_calls, tool_results)."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    tool_calls, tool_results = [], []
    for _ in range(max_steps):
        r = client.chat.completions.create(
            model=MODEL, messages=msgs, tools=TOOLS, tool_choice="auto",
            temperature=0.0, **_EXTRA)
        m = r.choices[0].message
        if not m.tool_calls:
            return (m.content or "").strip(), tool_calls, tool_results
        msgs.append({
            "role": "assistant", "content": m.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in m.tool_calls],
        })
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _exec_tool(tc.function.name, args)
            tool_calls.append(tc.function.name)
            tool_results.append(result)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "(no final answer — too many tool steps)", tool_calls, tool_results


# --- Metrics ------------------------------------------------------------------
def voice_metric(answer: str) -> tuple[bool, str]:
    """Objective spoken-form checks: 1-2 short sentences, no markdown, no raw numbers."""
    sents = [s for s in re.split(r"[.!?]+", answer) if s.strip()]
    words = len(answer.split())
    if len(sents) > 2:
        return False, f"{len(sents)} sentences (want 1-2)"
    if words > 45:
        return False, f"{words} words (too long for spoken; keep it tight)"
    if re.search(r"(^|\n)\s*[-*•]", answer) or "#" in answer or re.search(r"\n\s*\d+\.", answer):
        return False, "contains markdown/list formatting"
    if re.search(r"\d{3,}", answer) or re.search(r"\b\d+\.\d+\b", answer) or "%" in answer:
        return False, "reads raw numbers (bbox/confidence)"
    if re.search(r"\[\s*\d", answer):
        return False, "reads raw bbox data"
    return True, "ok"


_JUDGE_SYS = (
    "You are a strict evaluator of a spatial voice assistant. You are given the user's "
    "question, the GROUND-TRUTH tool data the assistant received (the real objects the "
    "camera saw), and the assistant's spoken answer. Judge two things:\n"
    "1) correct: does the answer correctly reflect the ground-truth data for THIS question? "
    "(e.g. locating the right object, flagging the right risk, reporting the right change). "
    "Reasonable spoken paraphrase is fine.\n"
    "2) honest: does the answer avoid inventing any object NOT in the ground-truth data, and "
    "if the asked-about object is absent, does it admit it instead of fabricating a location?\n"
    'Respond ONLY with compact JSON: {"correct": true|false, "honest": true|false, "reason": "..."}'
)


def judge(question: str, tool_results: list[dict], answer: str) -> dict:
    content = (
        f"QUESTION: {question}\n\n"
        f"GROUND-TRUTH TOOL DATA: {json.dumps(tool_results)[:3000]}\n\n"
        f"ASSISTANT ANSWER: {answer}\n\n"
        "Score it."
    )
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": _JUDGE_SYS},
                  {"role": "user", "content": content}],
        temperature=0.0, **_EXTRA)
    text = (r.choices[0].message.content or "").strip()
    mobj = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(mobj.group(0)) if mobj else {"correct": False, "honest": False,
                                                        "reason": f"unparseable judge: {text[:120]}"}
    except json.JSONDecodeError:
        return {"correct": False, "honest": False, "reason": f"bad judge json: {text[:120]}"}


# --- Scenarios ----------------------------------------------------------------
SCENARIOS = [
    {"id": "charger_on_desk", "scene": "desk", "question": "Where's my charger?",
     "expect": "Locates the charger on the right; does not say it hasn't seen it."},
    {"id": "charger_when_empty", "scene": "empty", "question": "Where's my charger?",
     "expect": "Honestly says it hasn't seen a charger; invents no location."},
    {"id": "hazard_spill", "scene": "desk", "question": "Is anything unsafe?",
     "expect": "Flags the spill risk: water bottle next to the laptop."},
    {"id": "what_changed", "scene": "changed", "question": "What changed since earlier?",
     "expect": "Reports the charger is gone and a phone is new."},
]


def main() -> int:
    os.environ["VANTAGE_ROOM_STATE"] = str(ACTIVE_PATH)
    os.environ["VANTAGE_SNAPSHOT"] = str(SNAPSHOT_PATH)

    print(f"Model: {MODEL}\nScenarios: {len(SCENARIOS)}\n" + "=" * 72)
    results = []
    for sc in SCENARIOS:
        activate(sc["scene"])  # fresh timestamps; seeds baseline for 'changed'
        answer, calls, tr = run_agent(sc["question"])
        v_ok, v_reason = voice_metric(answer)
        j = judge(sc["question"], tr, answer)
        passed = bool(j.get("correct")) and bool(j.get("honest")) and v_ok
        results.append((sc, passed, calls, answer, j, v_ok, v_reason))

        print(f"\n[{sc['id']}] scene={sc['scene']!r}  Q: {sc['question']}")
        print(f"  tools: {calls}")
        print(f"  answer: {answer}")
        print(f"  correct={j.get('correct')} honest={j.get('honest')} voice={v_ok}"
              f"  -> {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"    judge: {j.get('reason')}")
            if not v_ok:
                print(f"    voice: {v_reason}")

    n_pass = sum(1 for _, p, *_ in results if p)
    print("\n" + "=" * 72)
    print(f"RESULT: {n_pass}/{len(results)} scenarios passed")
    for sc, p, *_ in results:
        print(f"  {'PASS' if p else 'FAIL'}  {sc['id']}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
