"""Standalone Soniox TTS check — no LiveKit server, STT or LLM required.

Synthesizes the same sentence at several speeds and writes one WAV per speed,
so you can hear the `speed` parameter take effect.

Usage:
    export SONIOX_API_KEY=...
    python examples/voice_agents/soniox_tts_speed_check.py
"""

from __future__ import annotations

import asyncio
import pathlib
import wave

import aiohttp

from livekit.plugins import soniox

TEXT = "Hi there! This sentence is being spoken by Soniox text to speech."
SPEEDS = [0.7, 1.0, 1.3]
OUT_DIR = pathlib.Path(__file__).parent


async def synth_to_wav(session: aiohttp.ClientSession, speed: float) -> pathlib.Path:
    tts = soniox.TTS(voice="Maya", speed=speed, http_session=session)

    frames = []
    stream = tts.synthesize(TEXT)
    async for ev in stream:
        frames.append(ev.frame)
    await stream.aclose()
    await tts.aclose()

    out = OUT_DIR / f"soniox_speed_{speed}.wav"
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(frames[0].num_channels)
        wav.setsampwidth(2)  # pcm_s16le
        wav.setframerate(frames[0].sample_rate)
        for f in frames:
            wav.writeframes(f.data.tobytes())
    return out


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        for speed in SPEEDS:
            out = await synth_to_wav(session, speed)
            print(f"speed={speed} -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
