"""Consent-gated voice cloning: fetch a consented sample and register it at Cartesia.

The Platform never touches audio; it only sends the sample URL. This module downloads the
sample (https only, no private network targets) and uploads it to Cartesia's clone API.
Verified against https://docs.cartesia.ai/api-reference/voices/clone :
``POST https://api.cartesia.ai/voices/clone`` multipart form with ``clip`` (file, <=16 MB),
``name``, ``language`` (ISO 639-1), optional ``description``; response JSON ``{"id": ...}``.
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

CARTESIA_BASE_URL = "https://api.cartesia.ai"
# The clone endpoint documents a single supported version value (2026-08-14); the TTS
# preview/live path uses 2026-03-01. Change here if Cartesia rejects it.
CLONE_CARTESIA_VERSION = "2026-08-14"
SAMPLE_DOWNLOAD_TIMEOUT_SECONDS = 20.0
CLONE_UPLOAD_TIMEOUT_SECONDS = 60.0
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 3
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class UnsupportedSampleURL(Exception):
    """The sample URL is not an allowed public https URL."""


class SampleUnavailable(Exception):
    """The sample could not be downloaded or is unacceptable (type/size)."""


class CloneProviderError(Exception):
    """Cartesia call failed. The message is safe to log (no key, no upstream body)."""


async def _resolve_host(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise SampleUnavailable("sample host did not resolve") from exc
    return sorted({info[4][0] for info in infos})


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # is_global is False for private, loopback, link-local, reserved, unspecified and
    # shared (CGNAT 100.64.0.0/10) space, so one allow-list check covers every
    # non-routable range, including metadata addresses such as 100.100.100.200.
    # Multicast is not covered by is_global for IPv4, so exclude it explicitly.
    return ip.is_global and not ip.is_multicast


async def validate_sample_url(url: str) -> list[str]:
    """Return the validated public addresses for url, or raise UnsupportedSampleURL.

    url must be https, without credentials, to a host whose every address is globally
    routable. The caller must connect to one of the returned addresses (not re-resolve the
    hostname) so a DNS answer that changes after validation cannot redirect the request.
    """

    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError as exc:
        raise UnsupportedSampleURL("invalid sample url") from exc
    if parts.scheme != "https" or not host or parts.username or parts.password:
        raise UnsupportedSampleURL("sample url must be https without credentials")
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        addresses = await _resolve_host(host)
    if not addresses or not all(_is_public(address) for address in addresses):
        raise UnsupportedSampleURL("sample host is not a public address")
    return addresses


def _pinned_request(url: str, addresses: list[str]) -> tuple[str, dict[str, str], dict]:
    """Build (url, headers, extensions) that connect to a validated IP for url's host.

    TLS still verifies the certificate against the original hostname via the SNI
    extension, and the Host header keeps virtual hosting working.
    """

    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return url, {}, {}  # literal IP: nothing to pin, certificate is checked against it
    # Prefer IPv4 (stable ordering) over IPv6 when both were validated.
    ordered = sorted(addresses, key=lambda a: (":" in a, a))
    address = ordered[0].split("%", 1)[0]
    netloc = f"[{address}]" if ":" in address else address
    host_header = host
    if parts.port:
        netloc += f":{parts.port}"
        host_header += f":{parts.port}"
    pinned = parts._replace(netloc=netloc).geturl()
    return pinned, {"Host": host_header}, {"sni_hostname": host}


def _content_type_ok(content_type: str) -> bool:
    return content_type.startswith("audio/") or content_type == "application/octet-stream"


async def download_sample(url: str) -> tuple[bytes, str, str]:
    """Fetch the sample; returns (bytes, content_type, filename).

    Each hop is validated, then requested at the validated IP (no second DNS lookup).
    """

    current = url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            addresses = await validate_sample_url(current)
            pinned_url, headers, extensions = _pinned_request(current, addresses)
            # A fresh client per hop: the pool is keyed by IP, so reusing one across hops
            # could carry a connection verified for a different hostname.
            async with httpx.AsyncClient(
                timeout=SAMPLE_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=False
            ) as client:
                async with client.stream(
                    "GET", pinned_url, headers=headers, extensions=extensions
                ) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise SampleUnavailable("redirect without location")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise SampleUnavailable(
                            f"sample returned status {response.status_code}"
                        )
                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if not _content_type_ok(content_type):
                        raise SampleUnavailable("sample is not audio")
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > MAX_SAMPLE_BYTES:
                        raise SampleUnavailable("sample too large")
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_SAMPLE_BYTES:
                            raise SampleUnavailable("sample too large")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    if not data:
                        raise SampleUnavailable("sample is empty")
                    filename = urlsplit(current).path.rsplit("/", 1)[-1] or "sample"
                    return data, content_type, filename
        raise SampleUnavailable("too many redirects")
    except httpx.HTTPError as exc:
        raise SampleUnavailable(f"sample download failed: {type(exc).__name__}") from None


async def clone_voice(
    *,
    api_key: str,
    name: str,
    language: str,
    sample: bytes,
    content_type: str,
    filename: str,
    description: str | None = None,
) -> str:
    """Upload the sample to Cartesia and return the new provider voice id."""

    data = {"name": name, "language": language}
    if description:
        data["description"] = description
    headers = {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": CLONE_CARTESIA_VERSION,
    }
    try:
        async with httpx.AsyncClient(timeout=CLONE_UPLOAD_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{CARTESIA_BASE_URL}/voices/clone",
                headers=headers,
                data=data,
                files={"clip": (filename, sample, content_type)},
            )
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPStatusError as exc:
        raise CloneProviderError(
            f"cartesia returned status {exc.response.status_code}"
        ) from None
    except httpx.HTTPError as exc:
        raise CloneProviderError(f"cartesia request failed: {type(exc).__name__}") from None
    except ValueError:
        raise CloneProviderError("cartesia returned invalid json") from None
    voice_id = body.get("id") if isinstance(body, dict) else None
    if not isinstance(voice_id, str) or not voice_id:
        raise CloneProviderError("cartesia response missing voice id")
    return voice_id
