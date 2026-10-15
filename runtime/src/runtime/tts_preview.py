"""One-shot TTS preview synthesis through the Cartesia REST bytes endpoint."""

import httpx

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
