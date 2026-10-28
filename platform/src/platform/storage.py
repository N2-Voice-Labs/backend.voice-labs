"""Object-storage deletion seam for voice samples and previews.

Platform never stores audio itself; samples/previews live in object storage that
arrives with ST-31. Until then every call reports ``not_supported`` so callers can
record that the bytes were NOT confirmed deleted (e.g. leave ``sample_deleted_at``
unset) while still dropping their references.
"""

from typing import Literal

DeleteResult = Literal["deleted", "not_supported", "failed"]


def delete(uri: str | None) -> DeleteResult:
    """Delete one stored object by URI."""

    return "not_supported"


def delete_prefix(prefix: str) -> DeleteResult:
    """Delete every stored object under ``prefix``."""

    return "not_supported"
