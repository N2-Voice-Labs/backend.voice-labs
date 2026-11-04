"""scripts/render_apologies.py --elevenlabs: billing guards and the request it sends.

The script is loaded from its path (it is not a package). ``httpx.post`` is replaced in every
test, so nothing here reaches the network; the guard tests fail loudly if a request is attempted.
"""

import array
import importlib.util
import math
import wave
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_apologies.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("render_apologies_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ELEVENLABS_API_KEY", "CARTESIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class Calls(list):
    pass


@pytest.fixture
def no_http(monkeypatch) -> Calls:
    calls = Calls()

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a billable request was attempted")

    monkeypatch.setattr(httpx, "post", forbidden)
    return calls


def tone(seconds: float = 0.2, rate: int = 24000) -> bytes:
    samples = array.array(
        "h", (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(seconds * rate)))
    )
    return samples.tobytes()


class Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def fake_http(monkeypatch):
    calls: list[dict] = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return Response(tone(rate=int(kwargs["params"]["output_format"].removeprefix("pcm_"))))

    monkeypatch.setattr(httpx, "post", post)
    return calls


def run(script, tmp_path, *args) -> int:
    return script.main(["--elevenlabs", "--out", str(tmp_path / "out"), *args])


def test_refuses_without_confirm_billable(script, tmp_path, no_http, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    assert run(script, tmp_path, "--voice-id", "v", "--languages", "en") == 2
    assert "--confirm-billable" in capsys.readouterr().err
    assert no_http == [] and not (tmp_path / "out").exists()


def test_refuses_without_api_key(script, tmp_path, no_http, capsys) -> None:
    assert run(script, tmp_path, "--confirm-billable", "--voice-id", "v", "--languages", "en") == 2
    assert "ELEVENLABS_API_KEY" in capsys.readouterr().err
    assert no_http == [] and not (tmp_path / "out").exists()


def test_refuses_without_a_voice_id(script, tmp_path, no_http, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    assert run(script, tmp_path, "--confirm-billable", "--languages", "en", "ru") == 2
    assert "No voice for en" in capsys.readouterr().err
    assert no_http == []


def test_a_missing_voice_for_a_later_language_is_caught_before_any_request(
    script, tmp_path, no_http, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    code = run(script, tmp_path, "--confirm-billable", "--voice", "en=v-en", "--languages", "en", "ru")
    assert code == 2
    assert "No voice for ru" in capsys.readouterr().err
    assert no_http == []  # no billable clip was rendered for en first


def test_unsupported_rate_is_a_clean_error_without_a_request(
    script, tmp_path, no_http, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    code = run(script, tmp_path, "--confirm-billable", "--voice-id", "v", "--rate", "12345", "--languages", "en")
    assert code == 2
    assert "12345" in capsys.readouterr().err
    assert no_http == []


def test_happy_path_writes_a_valid_wav_and_sends_the_expected_request(
    script, tmp_path, fake_http, monkeypatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    code = run(script, tmp_path, "--confirm-billable", "--voice-id", "voice-1", "--languages", "ko", "en")
    assert code == 0
    for language in ("ko", "en"):
        with wave.open(str(tmp_path / "out" / f"{language}.wav"), "rb") as handle:
            assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 24000)
            assert handle.getnframes() > 0
    first = fake_http[0]
    assert first["url"] == "https://api.elevenlabs.io/v1/text-to-speech/voice-1"
    assert first["params"] == {"output_format": "pcm_24000"}
    assert first["headers"] == {"xi-api-key": "el-key"}
    assert first["json"]["model_id"] == "eleven_flash_v2_5"
    assert first["json"]["language_code"] == "ko"  # flash_v2_5 takes (and lists) ko
    assert first["json"]["text"]


def test_per_language_voice_and_rate_are_used(script, tmp_path, fake_http, monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    code = run(
        script, tmp_path, "--confirm-billable", "--voice", "en=v-en", "--voice", "ru=v-ru",
        "--rate", "16000", "--languages", "en", "ru",
    )
    assert code == 0
    assert [c["url"].rsplit("/", 1)[1] for c in fake_http] == ["v-en", "v-ru"]
    assert {c["params"]["output_format"] for c in fake_http} == {"pcm_16000"}
    with wave.open(str(tmp_path / "out" / "ru.wav"), "rb") as handle:
        assert handle.getframerate() == 16000


@pytest.mark.parametrize("model", ["eleven_multilingual_v2", "eleven_flash_v2", "unknown_model"])
def test_language_code_is_not_sent_for_models_that_do_not_take_one(
    script, tmp_path, fake_http, monkeypatch, model
) -> None:
    monkeypatch.setattr(script, "ELEVENLABS_MODEL", model)
    script.render_elevenlabs("en", tmp_path / "en.wav", 24000, "voice-1", "el-key")
    assert fake_http[0]["json"]["model_id"] == model
    assert "language_code" not in fake_http[0]["json"]


def test_language_table_matches_the_runtime_and_pipecat_tables(script) -> None:
    from src.runtime import tts_preview

    assert script.ELEVENLABS_LANGUAGE_CODES["eleven_flash_v2_5"] == (
        tts_preview.ELEVENLABS_MODEL_LANGUAGES["eleven_flash_v2_5"]
    )
    # every language the apology clips use is covered by the default model
    assert set(script.APOLOGY_TEXTS) <= script.ELEVENLABS_LANGUAGE_CODES[script.ELEVENLABS_MODEL]


@pytest.mark.parametrize(
    "bad", ["../../v1/user", "a/b", "a?b=c", "%2e%2e", "", "x" * 129, "has space", "abc\n"]
)
def test_unsafe_voice_ids_exit_2_before_any_request(script, tmp_path, no_http, monkeypatch, capsys, bad) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    flat = ["--voice-id", bad] if bad else ["--voice", "en=" + bad]
    assert run(script, tmp_path, "--confirm-billable", *flat, "--languages", "en") == 2
    assert no_http == [] and not (tmp_path / "out").exists()
    # per-language form too
    assert run(script, tmp_path, "--confirm-billable", "--voice", f"en={bad}", "--languages", "en") == 2
    assert no_http == []


def test_valid_real_looking_voice_id_ends_the_request_url(script, tmp_path, fake_http, monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    assert run(script, tmp_path, "--confirm-billable", "--voice-id", "JBFqnCBsd6RMkjVDRZzb", "--languages", "en") == 0
    assert fake_http[0]["url"].endswith("/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb")
