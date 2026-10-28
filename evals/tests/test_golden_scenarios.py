import json
from pathlib import Path

import pytest

from src.evals.dataset import DEFAULT_MANIFEST, load_manifest
from src.evals.golden import scenarios as sc

yaml = pytest.importorskip("yaml", reason="shipped scenarios are YAML")


def doc(**overrides):
    base = {"id": "one", "mode": "agent", "audio": ["en-ru-01"], "agent_config": {"system_prompt": "x"}}
    base.update(overrides)
    return base


def interpreter_doc(**overrides):
    config = {"interpreter": {"languages": ["ru", "en"], "direction": "one_way"}}
    return doc(mode="interpreter", agent_config=config, **overrides)


def test_shipped_scenarios_load_and_are_report_only():
    scenarios = sc.load_scenarios()
    ids = {s.id for s in scenarios}
    assert ids == {"en_agent_simple", "ru_agent_reply_language", "en_ru_code_switch_agent",
                   "ko_en_interpreter_two_way", "ru_en_interpreter_accurate", "glossary_term",
                   "provider_down", "tool_call"}
    assert not any(s.enforce for s in scenarios)
    by_id = {s.id: s for s in scenarios}
    assert by_id["ru_en_interpreter_accurate"].expect.min_heard_ratio == 0.966
    assert len(by_id["ru_en_interpreter_accurate"].audio) == 30
    assert by_id["provider_down"].expect.expect_session_failed
    assert by_id["glossary_term"].expect.glossary_terms == ("вид на жительство",)
    assert by_id["tool_call"].requires_env == ("EVALS_TOOL_SERVER_URL",)


def test_shipped_scenarios_reference_real_dataset_items_and_valid_interpreter_config():
    known = {item["id"] for item in load_manifest(DEFAULT_MANIFEST)}
    for scenario in sc.load_scenarios():
        assert sc.unknown_dataset_ids(scenario, known) == [], scenario.id
        if scenario.mode == "interpreter":
            interpreter = scenario.agent_config["interpreter"]
            assert not (interpreter["direction"] == "auto" and "ko" in interpreter["languages"])


def test_minimal_scenario_defaults():
    scenario = sc.parse_scenario(doc())
    assert scenario.enforce is False
    assert scenario.expect == sc.Expect()
    assert scenario.audio == (sc.AudioRef("en-ru-01", 0),)
    assert not scenario.expects_failure


def test_full_expect_block():
    scenario = sc.parse_scenario(doc(expect={
        "reply_language": "ru", "min_replies": 2, "max_reply_latency_ms": 3000, "max_p95_ms": 2500,
        "min_heard_ratio": 0.9, "glossary_terms": ["a|b"], "expect_provider_error": True,
        "expect_session_failed": True, "tool_called": "t"}, enforce=True))
    assert scenario.enforce and scenario.expects_failure
    assert scenario.expect.glossary_terms == ("a|b",)
    assert scenario.expect.min_heard_ratio == 0.9


@pytest.mark.parametrize("bad, fragment", [
    (doc(id="Bad Id"), "id must match"),
    (doc(mode="phone"), "mode must be"),
    (doc(audio=[]), "audio must be"),
    (doc(audio=[3]), "audio entry"),
    (doc(extra=1), "unknown key: extra"),
    (doc(expect={"min_heard_ratio": 1.5}), "between 0 and 1"),
    (doc(expect={"reply_language": "de"}), "reply_language"),
    (doc(expect={"typo_key": 1}), "unknown expect key: typo_key"),
    (doc(expect={"max_p95_ms": -1}), "non-negative"),
    (doc(expect={"glossary_terms": "x"}), "glossary_terms"),
    (doc(enforce="yes"), "enforce"),
    (doc(two_person=True), "two_person needs mode interpreter"),
    (doc(agent_config={"interpreter": {"languages": ["ru", "en"]}}), "must not set"),
    (doc(mode="interpreter"), "need agent_config.interpreter"),
    (interpreter_doc(audio=[{"ref": "x", "speaker": 1}]), "speaker 1 needs two_person"),
    (doc(mode="interpreter", agent_config={"interpreter": {"languages": ["ko", "en"], "direction": "auto"}}),
     "invalid with ko"),
    (doc(mode="interpreter", agent_config={"interpreter": {"languages": ["ko"]}}), "two languages"),
])
def test_invalid_scenarios_are_rejected_with_a_reason(bad, fragment):
    with pytest.raises(sc.ScenarioError, match=fragment):
        sc.parse_scenario(bad, "f.yaml")


def test_default_direction_is_auto_so_ko_needs_explicit_one_way():
    config = {"interpreter": {"languages": ["ko", "en"]}}
    with pytest.raises(sc.ScenarioError, match="invalid with ko"):
        sc.parse_scenario(doc(mode="interpreter", agent_config=config))


def test_loads_json_and_yaml_and_rejects_duplicates(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(doc(id="a")))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(doc(id="b")))
    (tmp_path / "notes.txt").write_text("ignored")
    assert [s.id for s in sc.load_scenarios(tmp_path)] == ["a", "b"]
    (tmp_path / "c.json").write_text(json.dumps(doc(id="a")))
    with pytest.raises(sc.ScenarioError, match="duplicate scenario id a"):
        sc.load_scenarios(tmp_path)


def test_broken_file_and_empty_dir(tmp_path):
    (tmp_path / "x.json").write_text("{nope")
    with pytest.raises(sc.ScenarioError, match="cannot parse"):
        sc.load_scenarios(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(sc.ScenarioError, match="no scenario files"):
        sc.load_scenarios(empty)


def test_select_only():
    scenarios = [sc.parse_scenario(doc(id="a")), sc.parse_scenario(doc(id="b"))]
    assert [s.id for s in sc.select(scenarios, ["b"])] == ["b"]
    assert sc.select(scenarios, None) == scenarios
    with pytest.raises(sc.ScenarioError, match="unknown scenario id"):
        sc.select(scenarios, ["zzz"])


# -- audio resolution and skip logic ------------------------------------------------------------


def make(tmp_path, **overrides):
    return sc.parse_scenario(doc(**overrides), "s.json", tmp_path)


def test_dataset_audio_resolves_or_skips(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    scenario = make(tmp_path)
    assert "dataset audio not generated: en-ru-01" in sc.skip_reason(scenario, audio, {})
    (audio / "en-ru-01.wav").write_bytes(b"")
    assert sc.skip_reason(scenario, audio, {}) is None


def test_human_audio_needs_env_dir_and_file(tmp_path):
    scenario = make(tmp_path, audio=["human_v1:ru_001"])
    audio = tmp_path / "audio"
    assert "EVALS_HUMAN_AUDIO_DIR is not set" in sc.skip_reason(scenario, audio, {})
    human = tmp_path / "human"
    human.mkdir()
    env = {"EVALS_HUMAN_AUDIO_DIR": str(human)}
    assert "human audio not found" in sc.skip_reason(scenario, audio, env)
    (human / "ru_001.wav").write_bytes(b"")
    assert sc.skip_reason(scenario, audio, env) is None
    path, reason = sc.resolve_clip(scenario.audio[0], scenario, audio, env)
    assert path == human / "ru_001.wav" and reason is None


def test_wav_path_is_relative_to_the_scenario_file(tmp_path):
    scenario = make(tmp_path, audio=["clips/one.wav"])
    assert "audio file not found" in sc.skip_reason(scenario, tmp_path, {})
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "one.wav").write_bytes(b"")
    assert sc.skip_reason(scenario, tmp_path, {}) is None


def test_requires_env_wins_and_blank_counts_as_unset(tmp_path):
    scenario = make(tmp_path, requires_env=["EVALS_TOOL_SERVER_URL"])
    assert sc.skip_reason(scenario, tmp_path, {}) == "EVALS_TOOL_SERVER_URL is not set"
    assert sc.skip_reason(scenario, tmp_path, {"EVALS_TOOL_SERVER_URL": "  "}) == "EVALS_TOOL_SERVER_URL is not set"
    (tmp_path / "en-ru-01.wav").write_bytes(b"")
    assert sc.skip_reason(scenario, tmp_path, {"EVALS_TOOL_SERVER_URL": "http://x"}) is None


def test_unknown_dataset_ids_ignores_human_and_paths():
    scenario = sc.parse_scenario(doc(audio=["nope-01", "human_v1:x", "a/b.wav", "en-ru-01"]))
    assert sc.unknown_dataset_ids(scenario, {"en-ru-01"}) == ["nope-01"]
