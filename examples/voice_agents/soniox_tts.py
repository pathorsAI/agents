import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, soniox
from livekit.plugins.soniox.tts import MAX_SPEED, MIN_SPEED

logger = logging.getLogger("soniox-tts-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self, tts: soniox.TTS) -> None:
        super().__init__(
            instructions="Your name is Kelly. You interact with users via voice, "
            "so keep your responses concise and to the point. "
            "Do not use emojis, asterisks, markdown, or other special characters. "
            "If the user asks you to speak faster or slower, call the "
            "set_speaking_speed tool to adjust your voice."
        )
        # Keep a typed reference to the Soniox TTS so tools can retune it live.
        # session.tts is typed as the provider-agnostic base and does not expose
        # update_options, so we hold the concrete instance here.
        self._tts = tts

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")

    @function_tool
    async def set_speaking_speed(self, context: RunContext, speed: float) -> str:
        """Adjust how fast you speak, effective from your next reply.

        Args:
            speed: Target speaking rate. 1.0 is normal, lower is slower, higher
                is faster. Accepted range is 0.7 to 1.3.
        """
        clamped = max(MIN_SPEED, min(MAX_SPEED, speed))
        self._tts.update_options(speed=clamped)
        logger.info("updated Soniox TTS speed to %s", clamped)

        if clamped != speed:
            return f"Speaking speed set to {clamped}, the closest allowed value to {speed}."
        return f"Speaking speed set to {clamped}."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # Soniox TTS. `speed` controls the speaking rate (range 0.7-1.3, 1.0 is
    # normal) and can be retuned mid-session via update_options (see the
    # set_speaking_speed tool). Requires SONIOX_API_KEY in the environment.
    tts = soniox.TTS(voice="Maya", speed=1.0)

    session: AgentSession = AgentSession(
        # Direct provider plugins: no LiveKit Cloud credentials needed, each
        # reads its own key from the environment (DEEPGRAM_API_KEY,
        # OPENAI_API_KEY). Swap for any provider you have keys for.
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=tts,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent=MyAgent(tts),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
