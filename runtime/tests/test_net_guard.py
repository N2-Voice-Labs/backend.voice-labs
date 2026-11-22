import asyncio

import pytest

from src.runtime import net_guard

PUBLIC_IP = "93.184.216.34"


def validate(url, *, resolve=None, schemes=("https",)):
    async def go():
        async def resolver(host):
            return resolve(host)

        return await net_guard.validate_public_url(
            url, schemes=schemes, resolver=resolver if resolve else None
        )

    return asyncio.run(go())


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/x",
        "https://169.254.169.254/latest/meta-data",
        "https://10.1.2.3/x",
        "https://172.16.0.1/x",
        "https://192.168.1.1/x",
        "https://[::1]/x",
        "https://[fd00::1]/x",
        "https://[::ffff:127.0.0.1]/x",
        "https://user:pw@93.184.216.34/x",
        "http://93.184.216.34/x",  # scheme not allowed
        "https:///nohost",
    ],
)
def test_validate_public_url_rejects_unsafe_targets(url) -> None:
    with pytest.raises(net_guard.UnsafeURL):
        validate(url)


def test_validate_public_url_checks_every_resolved_address() -> None:
    with pytest.raises(net_guard.UnsafeURL):
        validate("https://mixed.example.com/", resolve=lambda host: [PUBLIC_IP, "10.0.0.5"])
    assert validate("https://ok.example.com/", resolve=lambda host: [PUBLIC_IP]) == [PUBLIC_IP]


def test_validate_public_url_allows_configured_schemes() -> None:
    assert validate("http://ok.example.com/", resolve=lambda h: [PUBLIC_IP], schemes=("http", "https")) == [
        PUBLIC_IP
    ]
