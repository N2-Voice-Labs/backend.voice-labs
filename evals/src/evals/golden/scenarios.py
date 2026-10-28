"""Scenario schema and loading for the golden-path suite.

A scenario is one YAML (or JSON) file in ``scenarios/golden/``. YAML needs PyYAML (present in
the workspace environment); JSON files always load.

Fields
------
``id``            unique, ``[a-z0-9_]+``
``mode``          ``agent`` | ``interpreter``
``description``   free text
``agent_config``  body of ``POST /agents`` (``name`` and ``mode`` are filled in by the runner).
                  Interpreter scenarios need an ``interpreter`` block.
``audio``         what the simulated caller plays, in order. Each entry is a dataset v1 item id
                  (``ru-en-03``), ``human_v1:<name>`` (a recording in ``$EVALS_HUMAN_AUDIO_DIR``),
                  or a wav path (relative to the scenario file); or an object
                  ``{ref: ..., speaker: 0|1}`` for two-person scenarios.
``two_person``    interpreter only: a second simulated participant (language ``languages[1]``)
                  listens; ``speaker`` picks who plays each clip.
``requires_env``  environment variables that must be set, otherwise the scenario is skipped.
``expect``        thresholds, see :class:`Expect`; every key is optional.
``enforce``       ``false`` (default): report only. ``true``: a failure makes the run exit 1.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "scenarios" / "golden"
HUMAN_AUDIO_ENV = "EVALS_HUMAN_AUDIO_DIR"
HUMAN_PREFIX = "human_v1:"
MODES = ("agent", "interpreter")
REPLY_LANGUAGES = ("en", "ru", "ko")
ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
SUFFIXES = (".yaml", ".yml", ".json")


class ScenarioError(ValueError):
    """A scenario file is malformed; the message names the file and every problem."""


@dataclass(frozen=True)
class Expect:
    reply_language: str | None = None  # language of every reply: en | ru | ko
    min_replies: int | None = None  # assistant transcripts (messages) at least this many
    max_reply_latency_ms: float | None = None  # worst reply, end of caller speech -> heard
    max_p95_ms: float | None = None  # p95 of the same latencies
    min_heard_ratio: float | None = None  # share of played utterances that got a heard reply
    glossary_terms: tuple[str, ...] = ()  # each must appear in the replies; ``a|b`` = alternatives
    expect_provider_error: bool = False  # a ProviderError event must be observed
    expect_session_failed: bool = False  # the session must fail (SessionFailed / failed status)
    tool_called: str | None = None  # a ToolCalled event with this tool name must be observed


@dataclass(frozen=True)
class AudioRef:
    ref: str
    speaker: int = 0


@dataclass(frozen=True)
class Scenario:
    id: str
    mode: str
    agent_config: dict[str, Any]
    audio: tuple[AudioRef, ...]
    expect: Expect = field(default_factory=Expect)
    enforce: bool = False
    description: str = ""
    two_person: bool = False
    requires_env: tuple[str, ...] = ()
    source: str = ""  # file the scenario was loaded from
    base_dir: Path | None = None

    @property
    def expects_failure(self) -> bool:
        return self.expect.expect_session_failed or self.expect.expect_provider_error

    @property
    def languages(self) -> list[str]:
        return list((self.agent_config.get("interpreter") or {}).get("languages") or [])


_EXPECT_NUMBERS = ("min_replies", "max_reply_latency_ms", "max_p95_ms", "min_heard_ratio")
_SCENARIO_KEYS = {"id", "mode", "description", "agent_config", "audio", "two_person",
                  "requires_env", "expect", "enforce"}


def parse_expect(raw: Any, problems: list[str]) -> Expect:
    if raw is None:
        return Expect()
    if not isinstance(raw, Mapping):
        problems.append("expect must be a mapping")
        return Expect()
    known = {f for f in Expect.__dataclass_fields__}
    for key in sorted(set(raw) - known):
        problems.append(f"unknown expect key: {key}")
    values: dict[str, Any] = {}
    for key in _EXPECT_NUMBERS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            problems.append(f"expect.{key} must be a non-negative number")
        elif key == "min_heard_ratio" and value > 1:
            problems.append("expect.min_heard_ratio must be between 0 and 1")
        else:
            values[key] = value
    language = raw.get("reply_language")
    if language is not None:
        if language not in REPLY_LANGUAGES:
            problems.append(f"expect.reply_language must be one of {', '.join(REPLY_LANGUAGES)}")
        else:
            values["reply_language"] = language
    terms = raw.get("glossary_terms")
    if terms is not None:
        if not isinstance(terms, list) or not all(isinstance(t, str) and t.strip() for t in terms):
            problems.append("expect.glossary_terms must be a list of non-empty strings")
        else:
            values["glossary_terms"] = tuple(terms)
    for key in ("expect_provider_error", "expect_session_failed"):
        value = raw.get(key, False)
        if not isinstance(value, bool):
            problems.append(f"expect.{key} must be true or false")
        else:
            values[key] = value
    tool = raw.get("tool_called")
    if tool is not None:
        if not isinstance(tool, str) or not tool.strip():
            problems.append("expect.tool_called must be a tool name")
        else:
            values["tool_called"] = tool
    return Expect(**values)


def parse_audio(raw: Any, problems: list[str]) -> tuple[AudioRef, ...]:
    if not isinstance(raw, list) or not raw:
        problems.append("audio must be a non-empty list")
        return ()
    refs = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            refs.append(AudioRef(entry.strip()))
        elif isinstance(entry, Mapping) and isinstance(entry.get("ref"), str) and entry["ref"].strip():
            speaker = entry.get("speaker", 0)
            if speaker not in (0, 1) or isinstance(speaker, bool):
                problems.append(f"audio speaker must be 0 or 1: {entry!r}")
                continue
            refs.append(AudioRef(entry["ref"].strip(), speaker))
        else:
            problems.append(f"audio entry must be a string or {{ref, speaker}}: {entry!r}")
    return tuple(refs)


def parse_scenario(data: Any, source: str = "", base_dir: Path | None = None) -> Scenario:
    """Validate one decoded scenario document; raises :class:`ScenarioError`."""

    problems: list[str] = []
    if not isinstance(data, Mapping):
        raise ScenarioError(f"{source}: a scenario must be a mapping")
    for key in sorted(set(data) - _SCENARIO_KEYS):
        problems.append(f"unknown key: {key}")
    scenario_id = data.get("id")
    if not isinstance(scenario_id, str) or not ID_PATTERN.match(scenario_id):
        problems.append("id must match [a-z0-9_]+")
        scenario_id = str(scenario_id)
    mode = data.get("mode")
    if mode not in MODES:
        problems.append(f"mode must be one of {', '.join(MODES)}")
    config = data.get("agent_config", {})
    if not isinstance(config, Mapping):
        problems.append("agent_config must be a mapping")
        config = {}
    config = dict(config)
    interpreter = config.get("interpreter")
    if mode == "interpreter":
        if not isinstance(interpreter, Mapping):
            problems.append("interpreter scenarios need agent_config.interpreter")
        else:
            languages = interpreter.get("languages")
            if not isinstance(languages, list) or len(languages) != 2:
                problems.append("agent_config.interpreter.languages must list two languages")
            elif interpreter.get("direction", "auto") == "auto" and "ko" in languages:
                problems.append("direction auto is invalid with ko (use one_way)")
    elif mode == "agent" and interpreter is not None:
        problems.append("agent scenarios must not set agent_config.interpreter")
    two_person = data.get("two_person", False)
    if not isinstance(two_person, bool):
        problems.append("two_person must be true or false")
    elif two_person and mode != "interpreter":
        problems.append("two_person needs mode interpreter")
    enforce = data.get("enforce", False)
    if not isinstance(enforce, bool):
        problems.append("enforce must be true or false")
        enforce = False
    requires = data.get("requires_env", [])
    if not isinstance(requires, list) or not all(isinstance(v, str) for v in requires):
        problems.append("requires_env must be a list of variable names")
        requires = []
    audio = parse_audio(data.get("audio"), problems)
    if not two_person and any(a.speaker for a in audio):
        problems.append("speaker 1 needs two_person: true")
    expect = parse_expect(data.get("expect"), problems)
    if problems:
        raise ScenarioError(f"{source or scenario_id}: " + "; ".join(problems))
    return Scenario(
        id=scenario_id, mode=mode, agent_config=config, audio=audio, expect=expect,
        enforce=enforce, description=str(data.get("description") or "").strip(),
        two_person=two_person, requires_env=tuple(requires), source=source, base_dir=base_dir,
    )


def read_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml  # noqa: PLC0415 - optional, only YAML files need it
    except ImportError as exc:
        raise ScenarioError(f"{path}: PyYAML is not installed; use the workspace Python or a .json scenario") from exc
    return yaml.safe_load(text)


def load_scenario(path: Path) -> Scenario:
    try:
        data = read_document(path)
    except ScenarioError:
        raise
    except Exception as exc:  # noqa: BLE001 - JSON / YAML syntax errors
        raise ScenarioError(f"{path}: cannot parse: {exc}") from exc
    return parse_scenario(data, str(path), path.parent)


def load_scenarios(location: Path | str = SCENARIO_DIR) -> list[Scenario]:
    """Every scenario in a directory (sorted by file name) or a single file; ids must be unique."""

    location = Path(location)
    files = [location] if location.is_file() else sorted(
        p for p in location.iterdir() if p.suffix in SUFFIXES)
    if not files:
        raise ScenarioError(f"{location}: no scenario files ({', '.join(SUFFIXES)})")
    scenarios, seen = [], {}
    for path in files:
        scenario = load_scenario(path)
        if scenario.id in seen:
            raise ScenarioError(f"{path}: duplicate scenario id {scenario.id} (also in {seen[scenario.id]})")
        seen[scenario.id] = path.name
        scenarios.append(scenario)
    return scenarios


def select(scenarios: list[Scenario], only: list[str] | None) -> list[Scenario]:
    if not only:
        return scenarios
    known = {s.id for s in scenarios}
    unknown = sorted(set(only) - known)
    if unknown:
        raise ScenarioError(f"unknown scenario id(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    return [s for s in scenarios if s.id in set(only)]


# -- audio resolution and skip logic ----------------------------------------------------------


def resolve_clip(ref: AudioRef, scenario: Scenario, audio_dir: Path,
                 env: Mapping[str, str] | None = None) -> tuple[Path | None, str | None]:
    """(path, None) when the clip exists, else (None, reason). Never raises."""

    env = os.environ if env is None else env
    name = ref.ref
    if name.startswith(HUMAN_PREFIX):
        directory = env.get(HUMAN_AUDIO_ENV, "").strip()
        if not directory:
            return None, f"{HUMAN_AUDIO_ENV} is not set (human_v1 recordings are kept outside git)"
        stem = name[len(HUMAN_PREFIX):]
        path = Path(directory).expanduser() / (stem if stem.endswith(".wav") else f"{stem}.wav")
        return (path, None) if path.is_file() else (None, f"human audio not found: {path}")
    if name.endswith(".wav") or "/" in name or os.sep in name:
        path = Path(name).expanduser()
        if not path.is_absolute() and scenario.base_dir is not None:
            path = scenario.base_dir / path
        return (path, None) if path.is_file() else (None, f"audio file not found: {path}")
    path = audio_dir / f"{name}.wav"
    return (path, None) if path.is_file() else (None, f"dataset audio not generated: {name} (python -m src.evals.generate_audio)")


def skip_reason(scenario: Scenario, audio_dir: Path, env: Mapping[str, str] | None = None) -> str | None:
    """Why the scenario cannot run here, or ``None``. First missing thing wins."""

    env = os.environ if env is None else env
    for name in scenario.requires_env:
        if not env.get(name, "").strip():
            return f"{name} is not set"
    for ref in scenario.audio:
        _, reason = resolve_clip(ref, scenario, audio_dir, env)
        if reason:
            return reason
    return None


def unknown_dataset_ids(scenario: Scenario, known_ids: set[str]) -> list[str]:
    """Dataset-style audio refs that are not in the manifest (a typo, not a missing file)."""

    return [a.ref for a in scenario.audio
            if not a.ref.startswith(HUMAN_PREFIX) and not a.ref.endswith(".wav")
            and "/" not in a.ref and a.ref not in known_ids]
