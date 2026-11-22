"""SSRF guard shared by every Runtime call to a tenant-controlled URL.

``validate_public_url`` resolves the host and accepts it only when every address is globally
routable. ``pinned_request`` then rewrites the request to connect to one of the validated
addresses, so a DNS answer that changes after validation (DNS rebinding) cannot redirect the
request. Used by voice-sample download (https only) and tenant HTTP tools (http and https).
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urlsplit


class UnsafeURL(Exception):
    """The URL is malformed, has the wrong scheme/credentials, or targets a non-public host."""


class HostUnresolvable(Exception):
    """The host name did not resolve."""


Resolver = Callable[[str], Awaitable[list[str]]]


async def resolve_host(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise HostUnresolvable("host did not resolve") from exc
    return sorted({info[4][0] for info in infos})


def is_public(address: str) -> bool:
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


async def validate_public_url(
    url: str,
    *,
    schemes: Iterable[str] = ("https",),
    resolver: Resolver | None = None,
) -> list[str]:
    """Return the validated public addresses for url, or raise UnsafeURL.

    url must use one of ``schemes``, carry no credentials, and name a host whose every
    address is globally routable. The caller must connect to one of the returned addresses
    (see :func:`pinned_request`) instead of re-resolving the hostname. ``resolver`` is
    looked up per call so callers can substitute their own. :class:`HostUnresolvable`
    propagates from the resolver.
    """

    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError as exc:
        raise UnsafeURL("invalid url") from exc
    if parts.scheme not in tuple(schemes) or not host or parts.username or parts.password:
        raise UnsafeURL("url must use an allowed scheme and carry no credentials")
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        addresses = await (resolver or resolve_host)(host)
    if not addresses or not all(is_public(address) for address in addresses):
        raise UnsafeURL("host is not a public address")
    return addresses


def pinned_request(url: str, addresses: list[str]) -> tuple[str, dict[str, str], dict]:
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
