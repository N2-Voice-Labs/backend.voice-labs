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

    settings["voice"] = config.cartesia_voice_id
    return CartesiaTTSService(
        api_key=config.cartesia_api_key,
        settings=CartesiaTTSService.Settings(**settings),
        **extra,
    )


register_stt(
    "deepgram",
    _deepgram_stt,
    settings={"model": str, "language": str, "interim_results": bool, "smart_format": bool},
    features=("streaming", "keyterms", "multi_language"),
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
register_tts(
    "cartesia",
    _cartesia_tts,
    settings={"model": str, "language": str},
    features=("streaming", "voice_clone", "transport_destination"),
)
