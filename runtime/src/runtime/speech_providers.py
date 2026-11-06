"""STT, LLM and TTS provider registries (NS-06), the speech counterpart of ``translation_engine``.

Pipelines ask for "the STT service", "the LLM service" or "the TTS service" of the session
instead of importing a vendor service. The provider name is the one Platform sends from the
agent's provider records (``providers["stt"]`` / ``["llm"]`` / ``["tts"]`` in the session
contract, i.e. ``ProviderConfig.provider``). A new provider is one factory and one
``register_*`` call, declaring the settings keys its provider record may carry::

    def _elevenlabs_tts(config, settings, **extra):
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        ...

    register_tts("elevenlabs", _elevenlabs_tts, settings={"model": str, "language": str})

``PipelineConfig.from_environment`` rejects a provider that is not registered (and a settings
key it did not declare) when the session starts, never mid-call.

A factory receives the :class:`~.pipeline.PipelineConfig` (credentials, voice) and the
final settings dict (the session's ``provider_settings`` section plus the per-call language),
imports its Pipecat service lazily and returns the service. TTS factories also receive extra
keyword arguments for the service constructor (e.g. ``transport_destination``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .capabilities import ProviderCapabilities, ProviderKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import PipelineConfig

STTFactory = Callable[..., Any]  # (config, settings[, keyterms=]) -> pipecat STT service
LLMFactory = Callable[..., Any]  # (config, settings) -> pipecat LLM service
TTSFactory = Callable[..., Any]  # (config, settings, **extra) -> pipecat TTS service
SettingTypes = Mapping[str, "type | tuple[type, ...]"]

_STT: dict[str, STTFactory] = {}
_LLM: dict[str, LLMFactory] = {}
_TTS: dict[str, TTSFactory] = {}
_REGISTRIES: dict[str, dict[str, Callable[..., Any]]] = {"stt": _STT, "llm": _LLM, "tts": _TTS}
# Per provider: the settings keys its provider record may carry (name -> type), and its
# capabilities (``ProviderCapabilities``, see ``provider_capabilities``).
_SETTING_TYPES: dict[tuple[str, str], SettingTypes] = {}
_CAPABILITIES: dict[tuple[str, str], ProviderCapabilities] = {}


def _register(
    kind: ProviderKind,
    name: str,
    factory: Callable[..., Any],
    settings: SettingTypes | None,
    features: Sequence[str],
) -> Callable[..., Any]:
    _REGISTRIES[kind][name] = factory
    _SETTING_TYPES[(kind, name)] = dict(settings or {})
    _CAPABILITIES[(kind, name)] = ProviderCapabilities(kind, name, frozenset(features))
    return factory


def register_stt(
    name: str,
    factory: STTFactory,
    *,
    settings: SettingTypes | None = None,
    features: Sequence[str] = (),
) -> STTFactory:
    """Make ``name`` selectable as the session's STT provider (replaces the same name)."""

    return _register("stt", name, factory, settings, features)


def register_llm(
    name: str,
    factory: LLMFactory,
    *,
    settings: SettingTypes | None = None,
    features: Sequence[str] = (),
) -> LLMFactory:
    """Make ``name`` selectable as the session's LLM provider (replaces the same name)."""

    return _register("llm", name, factory, settings, features)


def register_tts(
    name: str,
    factory: TTSFactory,
    *,
    settings: SettingTypes | None = None,
    features: Sequence[str] = (),
) -> TTSFactory:
    """Make ``name`` selectable as the session's TTS provider (replaces the same name)."""

    return _register("tts", name, factory, settings, features)


def stt_names() -> tuple[str, ...]:
    return tuple(_STT)


def llm_names() -> tuple[str, ...]:
    return tuple(_LLM)


def tts_names() -> tuple[str, ...]:
    return tuple(_TTS)


def provider_names(kind: str) -> tuple[str, ...]:
    """The registered provider names of a slot (empty for an unknown slot)."""

    return tuple(_REGISTRIES.get(kind, ()))


def provider_setting_types(kind: str, provider: str) -> SettingTypes:
    """The settings keys (and their types) ``provider``'s record may carry in ``kind``."""

    return _SETTING_TYPES.get((kind, provider), {})


def provider_capabilities(kind: str, provider: str) -> ProviderCapabilities:
    """Capability metadata of a registered provider; ``KeyError`` for an unknown one."""

    return _CAPABILITIES[(kind, provider)]


def _factory(kind: str, config: "PipelineConfig") -> Callable[..., Any]:
    provider = config.providers[kind]
    try:
        return _REGISTRIES[kind][provider]
    except KeyError:
        raise RuntimeError(f"unsupported {kind} provider: {provider}") from None


def build_stt(
    config: "PipelineConfig", language: str | None = None, keyterms: Sequence[str] = ()
) -> Any:
    """The STT service of ``config.providers["stt"]``; ``language`` overrides the settings'.

    ``keyterms`` (glossary terms, NS-08) are passed to the factory only when there are some
    and the provider declares the ``keyterms`` feature; others never see them.
    """

    factory = _factory("stt", config)
    settings = config.provider_settings.get("stt", {}).copy()
    if language is not None:
        settings["language"] = language
    if keyterms and provider_capabilities("stt", config.providers["stt"]).supports("keyterms"):
        return factory(config, settings, keyterms=list(keyterms))
    return factory(config, settings)


def build_llm(config: "PipelineConfig", system_instruction: str | None = None) -> Any:
    """The LLM service of ``config.providers["llm"]``, with the session's LLM settings.

    ``translation_model`` is consumed by the translators only and never reaches the service.
    ``system_instruction`` (the agent prompt, already composed by the pipeline) is added to
    the settings when given.
    """

    factory = _factory("llm", config)
    settings = config.provider_settings.get("llm", {}).copy()
    settings.pop("translation_model", None)
    if system_instruction is not None:
        settings["system_instruction"] = system_instruction
    return factory(config, settings)


def build_tts(config: "PipelineConfig", language: str | None = None, **extra: Any) -> Any:
    """The TTS service of ``config.providers["tts"]``, speaking the session's voice."""

    factory = _factory("tts", config)
    settings = config.provider_settings.get("tts", {}).copy()
    if language is not None:
        settings["language"] = language
    return factory(config, settings, **extra)


def _deepgram_stt(
    config: "PipelineConfig", settings: dict[str, Any], keyterms: list[str] | None = None
) -> Any:
    from pipecat.services.deepgram.stt import DeepgramSTTService

    if keyterms and "keyterm" not in settings:
        settings["keyterm"] = keyterms  # nova-3 vocabulary boosting

    return DeepgramSTTService(
        api_key=config.deepgram_api_key,
        settings=DeepgramSTTService.Settings(**settings),
    )


def _openai_llm(config: "PipelineConfig", settings: dict[str, Any]) -> Any:
    """OpenAI and any OpenAI-compatible endpoint (``OPENAI_BASE_URL``)."""

    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key=config.openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        settings=OpenAILLMService.Settings(**settings),
    )


def _cartesia_tts(config: "PipelineConfig", settings: dict[str, Any], **extra: Any) -> Any:
    from pipecat.services.cartesia.tts import CartesiaTTSService

    settings["voice"] = config.tts_voice_id
    return CartesiaTTSService(
        api_key=config.cartesia_api_key,
        settings=CartesiaTTSService.Settings(**settings),
        **extra,
    )


GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def gemini_base_url() -> str:
    """Gemini's OpenAI-compatible endpoint (``GEMINI_BASE_URL`` overrides it)."""

    return os.getenv("GEMINI_BASE_URL") or GEMINI_DEFAULT_BASE_URL


def _gemini_llm(config: "PipelineConfig", settings: dict[str, Any]) -> Any:
    """Gemini through its OpenAI-compatible endpoint (agent mode).

    No model is chosen here: it comes from the provider record's ``model`` setting or
    ``GEMINI_MODEL``; without one the session fails to start instead of guessing.
    """

    from pipecat.services.openai.llm import OpenAILLMService

    if not settings.get("model"):
        model = os.getenv("GEMINI_MODEL")
        if not model:
            raise RuntimeError(
                "gemini llm needs a model: set the provider record's model or GEMINI_MODEL"
            )
        settings["model"] = model
    return OpenAILLMService(
        api_key=config.gemini_api_key,
        base_url=gemini_base_url(),
        settings=OpenAILLMService.Settings(**settings),
    )


def translation_endpoint(config: "PipelineConfig") -> tuple[str, str | None]:
    """``(api_key, base_url)`` of the text translator for ``config.providers["llm"]``."""

    if config.providers.get("llm") == "gemini":
        return config.gemini_api_key, gemini_base_url()
    return config.openai_api_key, os.getenv("OPENAI_BASE_URL") or None


GOOGLE_TRANSLATE = "google-translate"


def _google_translate_llm(config: "PipelineConfig", settings: dict[str, Any]) -> Any:
    raise RuntimeError(
        "google-translate only translates text: use it for translation sessions, "
        "not as an agent LLM"
    )


def build_translator(config: "PipelineConfig", *, setting: str = "model", default: str | None = None):
    """``(translator, model name)`` for ``config.providers["llm"]``."""

    from .interpreter import FallbackTranslator, GoogleTranslateTranslator, OpenAITranslator

    if config.providers.get("llm") == GOOGLE_TRANSLATE:
        translator = GoogleTranslateTranslator(api_key=config.google_translate_api_key)
        # LLM backup (glossary sentences, outages): Gemini when its key and a model are set.
        backup_model = config.provider_settings.get("llm", {}).get(
            "translation_model"
        ) or os.getenv("GEMINI_TRANSLATION_MODEL")
        if config.gemini_api_key and backup_model:
            backup = OpenAITranslator(
                api_key=config.gemini_api_key, model=backup_model, base_url=gemini_base_url()
            )
            translator = FallbackTranslator(translator, backup)
        return translator, translator.model
    model = resolve_translation_model(config, setting=setting, default=default)
    key, url = translation_endpoint(config)
    return OpenAITranslator(api_key=key, model=model, base_url=url), model


def resolve_translation_model(
    config: "PipelineConfig", *, setting: str = "model", default: str | None = None
) -> str:
    """The translation model: the provider setting, else ``default`` (OpenAI only).

    Gemini has no built-in default on purpose (model choice is not locked in code): its
    model is the provider record's ``setting`` or ``GEMINI_TRANSLATION_MODEL``.
    """

    configured = config.provider_settings.get("llm", {}).get(setting)
    if configured:
        return configured
    if config.providers.get("llm") == "gemini":
        model = os.getenv("GEMINI_TRANSLATION_MODEL")
        if not model:
            raise RuntimeError(
                f"gemini translation needs a model: set the llm provider's {setting} "
                "or GEMINI_TRANSLATION_MODEL"
            )
        return model
    if default is None:
        raise RuntimeError("no translation model configured")
    return default


def translation_provider(config: "PipelineConfig") -> str:
    """Name translation failures are reported under (``openai-compatible`` is ``openai``)."""

    name = config.providers.get("llm") or "openai"
    return "openai" if name == "openai-compatible" else name


def _elevenlabs_stt_service_class() -> Any:
    """Pipecat's realtime ElevenLabs STT plus fast reporting of a rejected API key."""

    global _ELEVENLABS_STT_SERVICE
    if _ELEVENLABS_STT_SERVICE is None:
        from pipecat.frames.frames import ErrorFrame
        from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService

        from .provider_failures import classify_error

        class VoiceLabsElevenLabsSTTService(ElevenLabsRealtimeSTTService):
            async def _maybe_try_reconnect(self, error_message, report_error, error=None):
                if (
                    error is not None
                    and not self._disconnecting
                    and classify_error(exception=error) == "auth"
                ):
                    await report_error(
                        ErrorFrame(error_message, exception=error),
                        force_treat_as_permanent=True,
                    )
                    return False
                return await super()._maybe_try_reconnect(error_message, report_error, error)

        _ELEVENLABS_STT_SERVICE = VoiceLabsElevenLabsSTTService
    return _ELEVENLABS_STT_SERVICE


_ELEVENLABS_STT_SERVICE: Any = None


def _elevenlabs_stt(
    config: "PipelineConfig", settings: dict[str, Any], keyterms: list[str] | None = None
) -> Any:
    """ElevenLabs realtime speech-to-text (Pipecat ``ElevenLabsRealtimeSTTService``).

    The default manual commit strategy is kept: Pipecat's VAD commits each utterance, so
    transcriptions arrive ``finalized`` like the interpreter expects. Without a ``model``
    setting Pipecat's own default applies. ``multi`` (Deepgram's code-switching value)
    means automatic language detection here, i.e. no language.
    """

    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService

    if settings.get("language") == "multi":
        settings["language"] = None
    if keyterms and "keyterms" not in settings:
        settings["keyterms"] = keyterms
    return _elevenlabs_stt_service_class()(
        api_key=config.elevenlabs_api_key,
        settings=ElevenLabsRealtimeSTTService.Settings(**settings),
    )


ELEVENLABS_DEFAULT_MODEL = "eleven_flash_v2_5"


_ELEVENLABS_SERVICE: Any = None


def _elevenlabs_service_class() -> Any:
    """Pipecat's ``ElevenLabsTTSService`` plus fast reporting of a rejected API key.

    Observed against the real API (2026-11-05): ElevenLabs accepts the WebSocket handshake
    for an invalid key and then closes with ``1008 (policy violation) Invalid API key``.
    Pipecat only logs that close, reconnects, and after three quick failures reports a
    generic "connection failed 3 times" error without the exception, which Voice Labs would
    classify as ``other``. Here an auth-classified close is reported immediately, with the
    exception, as a permanent failure.
    """

    global _ELEVENLABS_SERVICE
    if _ELEVENLABS_SERVICE is None:
        from pipecat.frames.frames import ErrorFrame
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        from .provider_failures import classify_error

        class VoiceLabsElevenLabsTTSService(ElevenLabsTTSService):
            async def _maybe_try_reconnect(self, error_message, report_error, error=None):
                if (
                    error is not None
                    and not self._disconnecting
                    and classify_error(exception=error) == "auth"
                ):
                    await report_error(
                        ErrorFrame(error_message, exception=error),
                        force_treat_as_permanent=True,
                    )
                    return False
                return await super()._maybe_try_reconnect(error_message, report_error, error)

        _ELEVENLABS_SERVICE = VoiceLabsElevenLabsTTSService
    return _ELEVENLABS_SERVICE


def _elevenlabs_tts(config: "PipelineConfig", settings: dict[str, Any], **extra: Any) -> Any:
    """ElevenLabs' multi-context WebSocket streaming TTS (Pipecat ``ElevenLabsTTSService``).

    ``extra`` (e.g. ``transport_destination``) goes to the service constructor, which forwards
    it to ``TTSService``. ``language`` is applied by Pipecat only for models that take a
    ``language_code`` (flash_v2_5, turbo_v2_5, v3, v3_conversational) and only when the model
    lists that language; otherwise it is logged and dropped. A voice or language change at
    runtime (``TTSUpdateSettingsFrame``) reconnects the WebSocket (both are URL fields).
    """

    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    import re

    from .voice_cloning import PROVIDER_VOICE_ID_PATTERN

    settings.setdefault("model", ELEVENLABS_DEFAULT_MODEL)
    settings["voice"] = config.tts_voice_id
    # Pipecat interpolates the voice id unescaped into the WebSocket URL path.
    if not re.fullmatch(PROVIDER_VOICE_ID_PATTERN, settings["voice"] or ""):
        raise RuntimeError("invalid elevenlabs voice id")
    service_class = _elevenlabs_service_class()
    return service_class(
        api_key=config.tts_api_key,
        settings=ElevenLabsTTSService.Settings(**settings),
        **extra,
    )


register_stt(
    "deepgram",
    _deepgram_stt,
    settings={"model": str, "language": str, "interim_results": bool, "smart_format": bool},
    features=("streaming", "keyterms", "multi_language"),
)
register_stt(
    "elevenlabs",
    _elevenlabs_stt,
    settings={"model": str, "language": str},
    features=("streaming", "keyterms"),
)
for _name in ("openai", "openai-compatible"):
    register_llm(
        _name,
        _openai_llm,
        settings={
            "model": str,
            "temperature": (int, float),
            "max_tokens": int,
            "top_p": (int, float),
            # Agent + translator: model of the two translation legs (not the LLM's).
            "translation_model": str,
        },
        features=("streaming", "tools"),
    )
register_llm(
    "gemini",
    _gemini_llm,
    settings={
        "model": str,
        "temperature": (int, float),
        "max_tokens": int,
        "top_p": (int, float),
        "translation_model": str,
    },
    features=("streaming", "tools"),
)
register_llm(GOOGLE_TRANSLATE, _google_translate_llm, settings={"translation_model": str}, features=("translation",))
register_tts(
    "cartesia",
    _cartesia_tts,
    settings={"model": str, "language": str},
    features=("streaming", "voice_clone", "transport_destination"),
)
register_tts(
    "elevenlabs",
    _elevenlabs_tts,
    settings={"model": str, "language": str},
    # No "voice_clone": cloning/deleting ElevenLabs voices is not implemented in the runtime.
    features=("streaming", "transport_destination"),
)
