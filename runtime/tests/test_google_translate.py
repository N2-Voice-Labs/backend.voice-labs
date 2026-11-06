"""Google Cloud Translation as the text translator (``llm`` provider ``google-translate``)."""

import asyncio

import httpx
import pytest

pytest.importorskip("pipecat")

from src.runtime import provider_health
from src.runtime.interpreter import GoogleTranslateTranslator, build_translation_messages
from src.runtime.speech_providers import build_translator

from test_elevenlabs_stack import stack_config


def translator_with(handler) -> GoogleTranslateTranslator:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GoogleTranslateTranslator(api_key="gt-key", client=client)


def test_translates_with_the_pair_from_the_prompt_and_a_header_key() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200, json={"data": {"translations": [{"translatedText": "It&#39;s fine"}]}}
        )

    messages = build_translation_messages("괜찮아요", "ko", "en")
    text = asyncio.run(translator_with(handler)(messages))
    request = seen["request"]
    assert text == "It's fine"  # HTML entities are decoded
    assert request.headers["x-goog-api-key"] == "gt-key"
    assert "gt-key" not in str(request.url)  # never in the URL (logs, error messages)
    body = request.read().decode()
    assert '"source":"ko"' in body.replace(" ", "") and '"target":"en"' in body.replace(" ", "")
    assert "괜찮아요" in body or "\\uad1c" in body


def test_http_errors_raise_so_the_engine_reports_a_provider_failure() -> None:
    translator = translator_with(lambda request: httpx.Response(403, json={}))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(translator(build_translation_messages("hi", "en", "ko")))


def test_build_translator_selects_google_without_a_model_or_llm_key() -> None:
    config = stack_config(
        providers={"stt": "elevenlabs", "llm": "google-translate", "tts": "elevenlabs"},
        provider_settings={},
        gemini_api_key="",
        google_translate_api_key="gt-key",
    )
    translator, model = build_translator(config)
    assert isinstance(translator, GoogleTranslateTranslator)
    assert model == "google-translate"


def test_only_the_google_key_is_required(monkeypatch) -> None:
    from src.runtime.pipeline import PipelineConfig

    session = {"providers": {"stt": "elevenlabs", "llm": "google-translate", "tts": "elevenlabs"}}
    for name in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_TRANSLATE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "v")
    monkeypatch.setenv("LIVEKIT_URL", "ws://x")
    monkeypatch.setenv("RUNTIME_LIVEKIT_TOKEN", "t")
    monkeypatch.setenv("LIVEKIT_ROOM_NAME", "r")
    with pytest.raises(RuntimeError, match="GOOGLE_TRANSLATE_API_KEY") as error:
        PipelineConfig.from_environment(session)
    assert "OPENAI" not in str(error.value) and "GEMINI" not in str(error.value)
    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "gt")
    assert PipelineConfig.from_environment(session).llm_api_key == "gt"


def test_health_probe_is_registered() -> None:
    assert any(p.name == "google-translate" for p in provider_health._PROBES)


# ---- fallback to the LLM -------------------------------------------------------------------

from src.runtime.interpreter import FallbackTranslator  # noqa: E402
from src.runtime.glossary import parse_glossary  # noqa: E402


class Fake:
    def __init__(self, model, result=None, error=None):
        self.model, self.result, self.error, self.calls = model, result, error, 0

    async def warm_up(self):
        pass

    async def __call__(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        if isinstance(self.result, str) and self.model == "mt":
            return self.result
        return self._stream()

    async def _stream(self):
        yield self.result


class FakeLLM(Fake):
    def __call__(self, messages):  # async generator, like OpenAITranslator
        self.calls += 1

        async def gen():
            yield self.result

        return gen()


def run(translator, messages):
    async def go():
        return "".join([d async for d in translator(messages)])

    return asyncio.run(go())


def test_machine_translation_answers_when_it_works() -> None:
    mt, llm = Fake("mt", "mt text"), FakeLLM("llm", "llm text")
    assert run(FallbackTranslator(mt, llm), build_translation_messages("hi", "en", "ko")) == "mt text"
    assert llm.calls == 0


def test_a_failure_uses_the_llm_and_skips_the_primary_for_a_while() -> None:
    mt, llm = Fake("mt", error=RuntimeError("down")), FakeLLM("llm", "llm text")
    translator = FallbackTranslator(mt, llm)
    messages = build_translation_messages("hi", "en", "ko")
    assert run(translator, messages) == "llm text"
    assert run(translator, messages) == "llm text"
    assert (mt.calls, llm.calls) == (1, 2)  # cooldown: no second wait on the broken service


def test_empty_machine_translation_uses_the_llm() -> None:
    mt, llm = Fake("mt", "  "), FakeLLM("llm", "llm text")
    assert run(FallbackTranslator(mt, llm), build_translation_messages("hi", "en", "ko")) == "llm text"


def test_glossary_sentences_go_straight_to_the_llm() -> None:
    glossary = parse_glossary([{"source": "N2", "keep": True}])
    messages = build_translation_messages("N2 is here", "en", "ko", glossary=glossary)
    mt, llm = Fake("mt", "mt text"), FakeLLM("llm", "llm text")
    assert run(FallbackTranslator(mt, llm), messages) == "llm text"
    assert mt.calls == 0


def test_build_translator_adds_the_llm_backup_only_when_configured(monkeypatch) -> None:
    base = dict(
        providers={"stt": "elevenlabs", "llm": "google-translate", "tts": "elevenlabs"},
        google_translate_api_key="gt-key",
    )
    monkeypatch.delenv("GEMINI_TRANSLATION_MODEL", raising=False)
    plain, _ = build_translator(stack_config(provider_settings={}, gemini_api_key="gm", **base))
    assert isinstance(plain, GoogleTranslateTranslator)  # no model: no guess
    backed, _ = build_translator(
        stack_config(provider_settings={"llm": {"translation_model": "any"}}, gemini_api_key="gm", **base)
    )
    assert isinstance(backed, FallbackTranslator)
    keyless, _ = build_translator(
        stack_config(provider_settings={"llm": {"translation_model": "any"}}, gemini_api_key="", **base)
    )
    assert isinstance(keyless, GoogleTranslateTranslator)


def test_a_hung_request_hits_the_deadline_instead_of_stalling_the_sentence() -> None:
    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(hang))
    translator = GoogleTranslateTranslator(api_key="gt-key", client=client, timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(translator(build_translation_messages("hi", "en", "ko")))


def test_deadline_and_endpoint_can_be_set_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_TRANSLATE_TIMEOUT_S", "0.7")
    monkeypatch.setenv("GOOGLE_TRANSLATE_BASE_URL", "http://localhost:9000/v2/")
    translator = GoogleTranslateTranslator(api_key="gt-key")
    assert translator._timeout == 0.7
    assert translator.URL == "http://localhost:9000/v2"


def test_a_failing_backup_is_reported_under_its_own_name(caplog) -> None:
    mt = Fake("mt", error=RuntimeError("down"))
    llm = FakeLLM("llm", "x")

    def broken(messages):
        async def gen():
            raise ConnectionError("gemini down")
            yield ""

        return gen()

    llm.__class__ = type("Broken", (FakeLLM,), {"__call__": lambda self, messages: broken(messages)})
    translator = FallbackTranslator(mt, llm, primary_name="google-translate", secondary_name="gemini")
    with caplog.at_level("WARNING"):
        with pytest.raises(ConnectionError) as caught:
            run(translator, build_translation_messages("hi", "en", "ko"))
    assert caught.value.translator_provider == "gemini"  # not the session's "google-translate"
    text = caplog.text
    assert "Primary translator (google-translate) failed" in text
    assert "Fallback translator (gemini) failed" in text


def test_a_lost_utterance_gets_a_notice_for_the_speaker_only() -> None:
    from src.runtime.apology import speaker_plans
    from src.runtime.participants import ParticipantDirectory

    people = {"a": {"language": "ko"}, "b": {"language": "en"}}
    directory = ParticipantDirectory.from_mapping(("ko", "en"), people)
    plans = speaker_plans(object(), directory, "a")
    assert len(plans) == 1  # one clip, on a's own output only
    assert plans[0].clip.language == "ko"
    assert speaker_plans(object(), directory, "nobody") == []
    assert speaker_plans(object(), None, "a") == []  # no private output to speak on


def test_diagnostics_report_whether_a_request_opened_a_connection(monkeypatch, capsys):
    from src.runtime.interpreter import GoogleTranslateTranslator

    def handler(request):
        return httpx.Response(200, json={"data": {"translations": [{"translatedText": "hi"}]}})

    messages = [{"role": "system", "content": "Translate from Korean (ko) to English (en)."}, {"role": "user", "content": "안녕"}]

    async def call():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await GoogleTranslateTranslator(api_key="k", client=client)(messages)

    monkeypatch.delenv("RUNTIME_TIMELINE_DIAG", raising=False)
    assert asyncio.run(call()) == "hi"
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("RUNTIME_TIMELINE_DIAG", "1")
    assert asyncio.run(call()) == "hi"
    assert capsys.readouterr().err.startswith("google_request new_connection=False ms=")
