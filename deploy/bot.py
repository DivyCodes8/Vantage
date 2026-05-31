#
# Vantage — CLOUD eval copy of the room-sense bot (Pipecat Cloud + Cekura).
#
# This is a logic-identical copy of the local bot_roomsense.py. The ONLY
# differences are:
#   1. It reads a FIXTURE room_state (VANTAGE_ROOM_STATE defaults to the bundled
#      fixture_desk.json) instead of the live camera — there is no vision here.
#   2. It builds its transport via create_transport() so it accepts the Daily
#      WebRTC sessions Pipecat Cloud / Cekura create.
#
# The system prompt is imported from vantage.agent_prompt (the SAME module the
# local bot uses, vendored into this bundle by build.sh) so the two never drift.
# No vision deps (torch/ultralytics) — room_query is pure stdlib.
#

import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FIXTURES = _HERE / "vantage" / "fixtures"

# Read a fixture, not the live camera. Default to the bundled desk scene; a
# per-session body (Cekura "Agent Configuration JSON") or the VANTAGE_FIXTURE /
# VANTAGE_ROOM_STATE secret can select another bundled fixture (see _prepare_scene).
os.environ.setdefault("VANTAGE_ROOM_STATE", str(_FIXTURES / "fixture_desk.json"))

from dotenv import load_dotenv

load_dotenv(override=True)  # On Pipecat Cloud, secrets arrive as env vars.

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, FunctionCallResultProperties, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

# Vendored provided services (verbatim copies of server/*, bundled by build.sh).
from nemotron_llm import VLLMOpenAILLMService
from nvidia_stt import NVidiaWebSocketSTTService

# Vendored Vantage modules (pure stdlib).
from vantage.agent_prompt import SYSTEM_INSTRUCTION
from vantage.memory import room_query


async def run_bot(transport: BaseTransport):
    """Assemble and run the Vantage room-sense pipeline (mirrors bot_roomsense.py)."""
    logger.info(f"Starting Vantage eval bot; fixture={os.environ.get('VANTAGE_ROOM_STATE')}")

    async def query_room_state(params: FunctionCallParams, query: str) -> None:
        """Look up what the camera currently sees, or find a specific object.

        Args:
            query: Object name (e.g. "charger", "water bottle") or a general
                phrase like "what do you see".
        """
        result = room_query.query_room_state(query)
        logger.info(f"query_room_state({query!r}) -> {result}")
        await params.result_callback(result)

    async def what_changed(params: FunctionCallParams) -> None:
        """Report what newly appeared, went missing, or moved since you last checked."""
        result = room_query.what_changed()
        logger.info(f"what_changed() -> {result}")
        await params.result_callback(result)

    async def check_hazards(params: FunctionCallParams) -> None:
        """Check the room for possible spill/trip/fall risks. Returns CANDIDATE
        risks for you to evaluate, rank, and report."""
        result = room_query.check_hazards()
        logger.info(f"check_hazards() -> {result}")
        await params.result_callback(result)

    async def end_call(params: FunctionCallParams) -> None:
        """End the conversation. Only call this AFTER saying goodbye in the same turn."""
        logger.info("end_call invoked")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    tool_functions = [query_room_state, what_changed, check_hazards, end_call]
    tools = ToolsSchema(standard_tools=tool_functions)

    stt = NVidiaWebSocketSTTService(url=os.environ["NVIDIA_ASR_URL"], strip_interim_prefix=True)

    enable_thinking = os.getenv("NEMOTRON_ENABLE_THINKING", "false").lower() == "true"
    llm = VLLMOpenAILLMService(
        api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),
        base_url=os.environ["NEMOTRON_LLM_URL"],
        settings=VLLMOpenAILLMService.Settings(
            model=os.getenv("NEMOTRON_LLM_MODEL", "nvidia/nemotron-3-super"),
            system_instruction=SYSTEM_INSTRUCTION,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}},
        ),
    )

    tts = GradiumTTSService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumTTSService.Settings(
            voice=os.getenv("GRADIUM_VOICE_ID", "Eu9iL_CYe8N-Gkx_"),
        ),
    )

    for fn in tool_functions:
        llm.register_direct_function(fn)

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=FilterIncompleteUserTurnStrategies(),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({
            "role": "user",
            "content": (
                "The user just connected. Greet them in ONE short sentence: say you're "
                "Vantage, you can see their room through the camera, and they can ask what "
                "you see, where something is, what changed, or if anything looks unsafe."
            ),
        })
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


def _prepare_scene(runner_args) -> str:
    """Pick which fixture this session reads, so one deployed agent can serve all
    Cekura scenarios.

    Priority: per-session body {"fixture": "desk|empty|changed"} (Cekura's Agent
    Configuration JSON) > VANTAGE_FIXTURE env > default "desk". Also seeds the
    what_changed() baseline for the "changed" scene so it deterministically
    reports the charger missing + phone new.

    NOTE: assumes scenarios run one session at a time (min_agents=1, sequential),
    since it sets process-level env + a /tmp snapshot.
    """
    body = getattr(runner_args, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = None
    name = (body.get("fixture") if isinstance(body, dict) else None) \
        or os.environ.get("VANTAGE_FIXTURE") or "desk"

    fpath = _FIXTURES / f"fixture_{name}.json"
    if not fpath.exists():
        name, fpath = "desk", _FIXTURES / "fixture_desk.json"
    os.environ["VANTAGE_ROOM_STATE"] = str(fpath)

    # Seed (or clear) the change baseline. Use /tmp (always writable).
    snap = Path("/tmp/vantage_change_snapshot.json")
    os.environ["VANTAGE_SNAPSHOT"] = str(snap)
    if name == "changed":
        desk = json.loads((_FIXTURES / "fixture_desk.json").read_text())
        baseline = {"taken": 0.0, "objects": {o["label"]: {"region": o["region"]}
                                              for o in desk["objects"] if o.get("visible")}}
        snap.write_text(json.dumps(baseline))
    elif snap.exists():
        snap.unlink()
    logger.info(f"scene={name} fixture={fpath}")
    return name


async def bot(runner_args: RunnerArguments):
    """Pipecat Cloud entry point. Accepts Daily (cloud/Cekura) and SmallWebRTC."""
    _prepare_scene(runner_args)

    def _daily_params():
        from pipecat.transports.daily.transport import DailyParams

        return DailyParams(audio_in_enabled=True, audio_out_enabled=True)

    transport = await create_transport(
        runner_args,
        {
            "daily": _daily_params,
            "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        },
    )
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
