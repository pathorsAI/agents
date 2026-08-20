"""Tests for the realtime usage-metadata drain at session teardown.

Gemini Live reports token usage only at end-of-turn; when a call ends mid-turn
(e.g. the caller hangs up while the agent is speaking its first line) the usage
event is still in flight. The fix under test has two halves:

1. the Gemini plugin's ``RealtimeSession.drain_pending_metrics()`` waits (bounded)
   for the current generation's usageMetadata;
2. ``AgentActivity._close_session`` awaits ``drain_pending_metrics()`` BEFORE it
   detaches the ``metrics_collected`` listener, so the drained event still lands
   in ``session.usage``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from google.genai import types as genai_types

from livekit.agents import Agent, AgentSession, llm, utils
from livekit.agents.metrics import LLMModelUsage, RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.plugins.google.realtime import realtime_api

from .fake_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


def _make_gemini_session() -> realtime_api.RealtimeSession:
    """Build a hermetic plugin RealtimeSession: no client, no websocket, no tasks.

    Only the state ``drain_pending_metrics`` / ``_handle_usage_metadata`` touch is
    initialized; ``_main_atask`` is a placeholder task standing in for a live
    connection loop.
    """
    sess = realtime_api.RealtimeSession.__new__(realtime_api.RealtimeSession)
    fake_model = MagicMock()
    fake_model.label = "google.realtime.RealtimeModel"
    fake_model.model = "gemini-test"
    fake_model.provider = "google"
    llm.RealtimeSession.__init__(sess, fake_model)
    sess._msg_ch = utils.aio.Chan()
    sess._session_should_close = asyncio.Event()
    sess._usage_received_ev = asyncio.Event()
    sess._current_generation = None
    sess._rejected_tool_calls = 0
    sess._active_session = MagicMock()
    sess._main_atask = asyncio.create_task(asyncio.sleep(30))
    return sess


def _make_generation() -> realtime_api._ResponseGeneration:
    return realtime_api._ResponseGeneration(
        message_ch=utils.aio.Chan(),
        function_ch=utils.aio.Chan(),
        input_id="GI_test",
        response_id="GR_test",
        text_ch=utils.aio.Chan(),
        audio_ch=utils.aio.Chan(),
        _created_timestamp=time.time(),
    )


async def _cleanup(sess: realtime_api.RealtimeSession) -> None:
    sess._main_atask.cancel()
    try:
        await sess._main_atask
    except asyncio.CancelledError:
        pass


async def test_drain_noop_without_generation() -> None:
    sess = _make_gemini_session()
    try:
        started = time.monotonic()
        await sess.drain_pending_metrics()
        assert time.monotonic() - started < 0.5
    finally:
        await _cleanup(sess)


async def test_drain_noop_when_usage_already_received() -> None:
    sess = _make_gemini_session()
    try:
        gen = _make_generation()
        gen._usage_received = True
        sess._current_generation = gen
        started = time.monotonic()
        await sess.drain_pending_metrics()
        assert time.monotonic() - started < 0.5
    finally:
        await _cleanup(sess)


async def test_drain_noop_when_not_connected() -> None:
    sess = _make_gemini_session()
    try:
        sess._current_generation = _make_generation()
        sess._active_session = None  # never connected / already torn down
        started = time.monotonic()
        await sess.drain_pending_metrics()
        assert time.monotonic() - started < 0.5
    finally:
        await _cleanup(sess)


async def test_drain_waits_for_usage_metadata() -> None:
    """A late usageMetadata delivered during the drain window is still emitted."""
    sess = _make_gemini_session()
    try:
        gen = _make_generation()
        sess._current_generation = gen

        collected: list[RealtimeModelMetrics] = []
        sess.on("metrics_collected", collected.append)

        usage = genai_types.UsageMetadata(
            prompt_token_count=100, response_token_count=25, total_token_count=125
        )

        async def _deliver_late() -> None:
            await asyncio.sleep(0.05)
            sess._handle_usage_metadata(usage)

        deliver_task = asyncio.create_task(_deliver_late())
        await sess.drain_pending_metrics()
        await deliver_task

        assert gen._usage_received
        assert len(collected) == 1
        assert collected[0].input_tokens == 100
        assert collected[0].output_tokens == 25
    finally:
        await _cleanup(sess)


async def test_drain_times_out_when_usage_never_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(realtime_api, "lk_google_usage_drain_timeout", 0.1)
    sess = _make_gemini_session()
    try:
        sess._current_generation = _make_generation()
        started = time.monotonic()
        await sess.drain_pending_metrics()
        elapsed = time.monotonic() - started
        assert 0.05 < elapsed < 1.0
    finally:
        await _cleanup(sess)


async def test_drain_disabled_via_timeout() -> None:
    sess = _make_gemini_session()
    try:
        sess._current_generation = _make_generation()
        orig = realtime_api.lk_google_usage_drain_timeout
        realtime_api.lk_google_usage_drain_timeout = 0.0
        try:
            started = time.monotonic()
            await sess.drain_pending_metrics()
            assert time.monotonic() - started < 0.5
        finally:
            realtime_api.lk_google_usage_drain_timeout = orig
    finally:
        await _cleanup(sess)


async def test_usage_metadata_marks_generation_and_event() -> None:
    sess = _make_gemini_session()
    try:
        gen = _make_generation()
        sess._current_generation = gen
        assert not sess._usage_received_ev.is_set()

        sess._handle_usage_metadata(
            genai_types.UsageMetadata(
                prompt_token_count=10, response_token_count=5, total_token_count=15
            )
        )
        assert gen._usage_received
        assert sess._usage_received_ev.is_set()
    finally:
        await _cleanup(sess)


async def test_close_session_drains_before_detaching_metrics_listener() -> None:
    """AgentSession.aclose() must collect a usage event flushed by the drain hook.

    This is the ordering the whole fix depends on: _close_session awaits
    drain_pending_metrics() BEFORE off("metrics_collected"), so a usage event
    emitted from the drain still reaches session.usage.
    """
    model = FakeRealtimeModel()
    session: AgentSession = AgentSession(llm=model)
    await session.start(Agent(instructions="test"))

    rt = model.active_session
    metric = RealtimeModelMetrics(
        label=model.label,
        request_id="GR_last_turn",
        timestamp=time.time(),
        input_tokens=111,
        output_tokens=22,
        total_tokens=133,
        input_token_details=RealtimeModelMetrics.InputTokenDetails(audio_tokens=90, text_tokens=10),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(audio_tokens=22),
        metadata=Metadata(model_name="fake-realtime", model_provider="fake"),
    )

    drained = asyncio.Event()

    async def _drain_pending_metrics() -> None:
        # simulate the provider flushing the in-flight usage event during the grace
        rt.emit("metrics_collected", metric)
        drained.set()

    rt.drain_pending_metrics = _drain_pending_metrics  # type: ignore[method-assign]

    await session.aclose()

    assert drained.is_set()
    llm_usage = [u for u in session.usage.model_usage if isinstance(u, LLMModelUsage)]
    assert sum(u.input_tokens for u in llm_usage) == 111
    assert sum(u.output_tokens for u in llm_usage) == 22
