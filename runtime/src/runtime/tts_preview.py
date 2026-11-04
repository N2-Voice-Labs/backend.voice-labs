"""One-shot TTS preview synthesis (REST), selected by provider name (NS-06).

``PREVIEW_PROVIDERS`` is the preview counterpart of the TTS registry in
``speech_providers``: ``/tts/preview`` looks the provider up here instead of naming Cartesia.
A new provider is one ``register_preview(name, api_key_env, synthesize)`` call.
"""

import io
import re
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from .voice_cloning import PROVIDER_VOICE_ID_PATTERN

CARTESIA_BASE_URL = "https://api.cartesia.ai"
# Match the values pipecat's CartesiaTTSService uses for live calls
# (pipecat/services/cartesia/tts.py: _CARTESIA_API_VERSION and default model).
CARTESIA_VERSION = "2026-03-01"
DEFAULT_MODEL = "sonic-3.6"
PREVIEW_TIMEOUT_SECONDS = 15.0
PREVIEW_SAMPLE_RATE = 24000


class TTSPreviewError(Exception):
    """Provider call failed. The message is safe to log (no key, no upstream body)."""


async def synthesize_preview(
    *,
    api_key: str,
    provider_voice_id: str,
    text: str,
    language: str | None = None,
    model: str | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "model_id": model or DEFAULT_MODEL,
        "transcript": text,
        "voice": {"mode": "id", "id": provider_voice_id},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": PREVIEW_SAMPLE_RATE,
        },
    }
    if language:
        payload["language"] = language
    headers = {
        "X-API-Key": api_key,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=PREVIEW_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{CARTESIA_BASE_URL}/tts/bytes", headers=headers, json=payload
            )
            response.raise_for_status()
            audio = response.content
    except httpx.HTTPStatusError as exc:
        raise TTSPreviewError(
            f"cartesia returned status {exc.response.status_code}"
        ) from None
    except httpx.HTTPError as exc:
        raise TTSPreviewError(f"cartesia request failed: {type(exc).__name__}") from None
    if not audio:
        raise TTSPreviewError("cartesia returned empty audio")
    return audio


ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
# Flash is ElevenLabs' low-latency, lower-cost multilingual model; override per request.
ELEVENLABS_DEFAULT_MODEL = "eleven_flash_v2_5"


# Models ElevenLabs accepts ``language_code`` for, with the languages each covers. This
# mirrors ``pipecat.services.elevenlabs.tts_base.ELEVENLABS_MODEL_LANGUAGES`` (Pipecat 1.12.0)
# on purpose: the preview must work without the voice extra, so Pipecat is not imported here
# (a test compares the two). ``eleven_multilingual_v2`` and unknown models are absent: the
# REST reference documents ``language_code`` as unsupported for multilingual_v2, so nothing is sent.
_ELEVENLABS_V2_5_LANGUAGES = frozenset(
    """
    ar bg cs da de el en es fi fil fr hi hr hu id it
    ja ko ms nl no pl pt ro ru sk sv ta tr uk vi zh
    """.split()
)
_ELEVENLABS_V3_LANGUAGES = frozenset(
    """
    af ar as az be bg bn bs ca ceb cs cy da de el en
    es et fa fi fil fr ga gl gu ha he hi hr hu hy id
    is it ja jv ka kk kn ko ky lb ln lt lv mk ml mr
    ms ne nl no ny pa pl ps pt ro ru sd sk sl so sr
    sv sw ta te th tr uk ur vi zh
    """.split()
)
ELEVENLABS_MODEL_LANGUAGES: dict[str, frozenset[str]] = {
    "eleven_flash_v2_5": _ELEVENLABS_V2_5_LANGUAGES,
    "eleven_turbo_v2_5": _ELEVENLABS_V2_5_LANGUAGES,
    "eleven_v3": _ELEVENLABS_V3_LANGUAGES,
    "eleven_v3_conversational": _ELEVENLABS_V3_LANGUAGES,
}


def elevenlabs_language_code(model: str | None, language: str | None) -> str | None:
    """The ``language_code`` to send for ``model``, or None (send nothing).

    A region suffix (``ko-KR``) is reduced to its base code. Returns None for a model that
    takes no language code (``eleven_multilingual_v2``, unknown models) and for a language
    the model does not list.
    """

    supported = ELEVENLABS_MODEL_LANGUAGES.get(model or "")
    if supported is None or not language:
        return None
    base = language.strip().lower().replace("_", "-").split("-", 1)[0]
    return base if base in supported else None


async def synthesize_elevenlabs_preview(
    *,
    api_key: str,
    provider_voice_id: str,
    text: str,
    language: str | None = None,
    model: str | None = None,
) -> bytes:
    """One-shot ElevenLabs synthesis returned as a mono 16-bit WAV (like the Cartesia one).

    ``language_code`` is sent only when the model supports it and lists the language
    (see :func:`elevenlabs_language_code`).

    Caveat: the ElevenLabs docs do not state that ``pcm_<rate>`` responses are headerless
    16-bit little-endian mono samples. That is assumed from how Pipecat's ElevenLabs service
    feeds ``pcm_<rate>`` bytes straight into its audio frames, and has not been verified
    against the live API here. If a response were not raw PCM, the WAV would be noise.
    """

    # The id goes into the URL path with the operator's API key attached: reject anything
    # that could alter the path ("../", "?", "/", whitespace) before any request, and quote
    # it as defense in depth.
    if not isinstance(provider_voice_id, str) or not re.fullmatch(
        PROVIDER_VOICE_ID_PATTERN, provider_voice_id
    ):
        raise TTSPreviewError("invalid voice id")
    chosen_model = model or ELEVENLABS_DEFAULT_MODEL
    payload: dict[str, object] = {"text": text, "model_id": chosen_model}
    language_code = elevenlabs_language_code(chosen_model, language)
    if language_code:
        payload["language_code"] = language_code
    try:
        async with httpx.AsyncClient(timeout=PREVIEW_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{ELEVENLABS_BASE_URL}/v1/text-to-speech/{quote(provider_voice_id, safe='')}",
                params={"output_format": f"pcm_{PREVIEW_SAMPLE_RATE}"},
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            pcm = response.content
    except httpx.HTTPStatusError as exc:
        raise TTSPreviewError(
            f"elevenlabs returned status {exc.response.status_code}"
        ) from None
    except httpx.HTTPError as exc:
        raise TTSPreviewError(f"elevenlabs request failed: {type(exc).__name__}") from None
    if not pcm:
        raise TTSPreviewError("elevenlabs returned empty audio")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(PREVIEW_SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True)
class PreviewProvider:
    api_key_env: str
    # async (*, api_key, provider_voice_id, text, language, model) -> WAV bytes
    synthesize: Callable[..., Awaitable[bytes]]


PREVIEW_PROVIDERS: dict[str, PreviewProvider] = {}


def register_preview(
    name: str, api_key_env: str, synthesize: Callable[..., Awaitable[bytes]]
) -> None:
    """Make ``name`` previewable (replaces the same name)."""

    PREVIEW_PROVIDERS[name] = PreviewProvider(api_key_env, synthesize)


def get_preview_provider(name: str) -> PreviewProvider:
    try:
        return PREVIEW_PROVIDERS[name]
    except KeyError:
        raise TTSPreviewError(f"unsupported tts provider: {name}") from None


register_preview("cartesia", "CARTESIA_API_KEY", synthesize_preview)
register_preview("elevenlabs", "ELEVENLABS_API_KEY", synthesize_elevenlabs_preview)
