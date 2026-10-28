"""Run golden-path scenarios against a live Platform + Runtime + LiveKit stack.

Reuses :mod:`evals.live_latency` (Platform client, fake microphone, voice detection). A run
is split so that everything except the audio itself is testable offline:

* :func:`run_scenario` drives the Platform API (agent, session, participants, transcripts,
  runtime state) and turns what happened into an :class:`~.checks.Observation`;
* a *driver* plays the caller audio and reports what was heard. :class:`LivekitDriver` is
  the real one (billable); tests pass a fake.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..live_latency import (
    FRAME_S,
    JOIN_WAIT_S,
    SETTLE_S,
    Person,
    Platform,
    read_wav,
    speech_bounds,
    wait_for_translation,
)
from .checks import PROVIDER_ERROR_EVENTS, CheckResult, Observation, run_checks
from .scenarios import Scenario, resolve_clip, skip_reason

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"


@dataclass(frozen=True)
class Settings:
    pause: float = 2.0  # silence between utterances
    wait: float = 15.0  # seconds after the end of speech before a reply counts as missing
    settle: float | None = None  # silence that ends a reply (default: SETTLE_S per engine)
    event_wait: float = 15.0  # seconds to wait after the call for transcripts and events
    join_wait: float = JOIN_WAIT_S
    poll_interval: float = 1.0


@dataclass(frozen=True)
class Clip:
    ref: str
    path: Path
    speaker: int = 0


@dataclass(frozen=True)
class Heard:
    """What the listener heard after one clip; ``after_end_s`` is onset minus end of caller speech."""

    ref: str
    heard: bool
    after_end_s: float | None = None


@dataclass
class ScenarioResult:
    id: str
    mode: str
    status: str
    enforce: bool = False
    reason: str = ""  # why skipped, or the error that stopped the run
    checks: list[CheckResult] = field(default_factory=list)
    observation: Observation | None = None
    duration_s: float = 0.0
    agent_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        obs = self.observation
        return {
            "id": self.id, "mode": self.mode, "status": self.status, "enforce": self.enforce,
            "reason": self.reason, "duration_s": round(self.duration_s, 2),
            "checks": [c.to_dict() for c in self.checks],
            "observation": None if obs is None else {
                "played": obs.played, "heard": obs.heard,
                "heard_latencies_ms": [round(v, 1) for v in obs.heard_latencies_ms],
                "replies": obs.replies, "event_types": obs.event_types,
                "session_status": obs.session_status, "session_error": obs.session_error,
            },
            "agent_id": self.agent_id, "session_id": self.session_id,
            "conversation_id": self.conversation_id,
        }


class Driver(Protocol):
    async def play(self, scenario: Scenario, connections: list[tuple[str, dict[str, str]]],
                   clips: list[Clip], settings: Settings) -> list[Heard]: ...


# -- Platform ----------------------------------------------------------------------------------


class GoldenPlatform(Platform):
    """:class:`~evals.live_latency.Platform` plus the calls agent scenarios need."""

    async def create_agent_body(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._call("POST", "/agents", body)

    async def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return await self._call("GET", f"/conversations/{conversation_id}/messages") or []

    async def runtime_state(self, conversation_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/conversations/{conversation_id}/runtime-state") or {}


# -- live driver ---------------------------------------------------------------------------------


class LivekitDriver:
    """Joins the room with fake microphones (billable: the Runtime calls paid providers)."""

    async def play(self, scenario: Scenario, connections: list[tuple[str, dict[str, str]]],
                   clips: list[Clip], settings: Settings) -> list[Heard]:
        from livekit import rtc  # noqa: PLC0415 - only needed for live runs

        engine = (scenario.agent_config.get("interpreter") or {}).get("engine", "cascaded")
        settle = settings.settle if settings.settle is not None else SETTLE_S.get(engine, 1.2)
        people = [Person(rtc, label, connection, False) for label, connection in connections]
        heard: list[Heard] = []
        try:
            for person in people:
                await person.join()
            await asyncio.sleep(settings.join_wait)
            for clip in clips:
                speaker = people[clip.speaker if len(people) > 1 else 0]
                listener = people[1 - clip.speaker] if len(people) > 1 else people[0]
                pcm = read_wav(clip.path)
                lead, trail = speech_bounds(pcm)
                first = speaker.mic.say(pcm)
                start = speaker.mic.time_of(first + round(lead / FRAME_S))
                end = speaker.mic.time_of(first + round(trail / FRAME_S))
                await asyncio.sleep(max(0.0, end - time.monotonic()))
                await wait_for_translation(listener, start, end, settings.wait, settle)
                onset = listener.detector.first_onset_since(start)
                heard.append(Heard(clip.ref, onset is not None, None if onset is None else onset - end))
                await asyncio.sleep(settings.pause)
        finally:
            for person in people:
                try:
                    await person.leave()
                except Exception:  # noqa: BLE001
                    pass
        return heard


# -- scenario run ---------------------------------------------------------------------------------


def agent_body(scenario: Scenario, run_id: str) -> dict[str, Any]:
    """``POST /agents`` body: the scenario's config with a recognisable name and its mode."""

    return {**scenario.agent_config, "name": f"golden {scenario.id} {run_id}", "mode": scenario.mode}


async def collect_observation(
    platform: GoldenPlatform, conversation_id: str, scenario: Scenario, heard: list[Heard],
    played: int, settings: Settings, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> Observation:
    """Transcripts and runtime events arrive asynchronously; poll until they look complete."""

    obs = Observation(played=played, heard_latencies_ms=[
        h.after_end_s * 1000 for h in heard if h.heard and h.after_end_s is not None])
    expect = scenario.expect
    want = expect.min_replies if expect.min_replies is not None else (0 if scenario.expects_failure else 1)
    attempts = int(settings.event_wait / settings.poll_interval) + 1
    for attempt in range(attempts):
        obs.replies = [m["content"] for m in await platform.messages(conversation_id)
                       if m.get("role") == "assistant" and m.get("content")]
        state = await platform.runtime_state(conversation_id)
        obs.events = list(state.get("events") or [])
        obs.session_status = state.get("session_status")
        types = set(obs.event_types)
        done = len(obs.replies) >= want
        if expect.expect_provider_error:
            done = done and bool(types & PROVIDER_ERROR_EVENTS)
        if expect.expect_session_failed:
            done = done and (obs.session_status == "failed" or bool(types & {"SessionFailed", "RuntimeFailed"}))
        if expect.tool_called:
            done = done and "ToolCalled" in types
        if done or attempt == attempts - 1:
            break
        await sleep(settings.poll_interval)
    return obs


async def run_scenario(
    scenario: Scenario, platform: GoldenPlatform, driver: Driver, audio_dir: Path, settings: Settings,
    run_id: str, env: Mapping[str, str] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> ScenarioResult:
    """Run one scenario end to end; never raises (errors become a failed result)."""

    result = ScenarioResult(scenario.id, scenario.mode, FAILED, scenario.enforce)
    reason = skip_reason(scenario, audio_dir, env)
    if reason:
        result.status, result.reason = SKIPPED, reason
        return result

    started = time.monotonic()
    refs = scenario.audio[:1] if scenario.expects_failure else scenario.audio
    clips = [Clip(a.ref, resolve_clip(a, scenario, audio_dir, env)[0], a.speaker) for a in refs]
    heard: list[Heard] = []
    obs = Observation(played=len(clips))
    error: str | None = None
    try:
        agent = await platform.create_agent_body(agent_body(scenario, run_id))
        result.agent_id = agent["id"]
        languages = scenario.languages
        try:
            session = await platform.create_session(
                result.agent_id, languages[0] if scenario.two_person else None)
        except RuntimeError as exc:  # e.g. 503 runtime_unavailable: the signal provider_down looks for
            obs.session_error = str(exc)
            session = None
        if session is not None:
            result.session_id, result.conversation_id = session["session_id"], session.get("conversation_id")
            connections = [(f"A-{languages[0]}" if languages else "A", session["connection"])]
            if scenario.two_person:
                guest = await platform.add_participant(result.session_id, languages[1])
                connections.append((f"B-{languages[1]}", guest["connection"]))
            try:
                heard = await driver.play(scenario, connections, clips, settings)
            except Exception as exc:  # noqa: BLE001 - keep what the Platform saw
                error = f"playing audio: {type(exc).__name__}: {exc}"
            finally:
                try:
                    await platform.end_session(result.session_id)
                except Exception as exc:  # noqa: BLE001
                    error = error or f"end session: {exc}"
            if result.conversation_id:
                obs = await collect_observation(
                    platform, result.conversation_id, scenario, heard, len(clips), settings, sleep)
    except Exception as exc:  # noqa: BLE001 - one broken scenario must not lose the others
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if result.agent_id:
            try:
                await platform.disable_agent(result.agent_id)
            except Exception:  # noqa: BLE001 - cleanup only
                pass

    result.observation = obs
    result.checks = run_checks(scenario.expect, obs)
    if error and not scenario.expects_failure:
        result.checks.insert(0, CheckResult("run_completed", False, error, "no error"))
        result.reason = error
    result.status = PASSED if result.checks and all(c.passed for c in result.checks) else FAILED
    result.duration_s = time.monotonic() - started
    return result
