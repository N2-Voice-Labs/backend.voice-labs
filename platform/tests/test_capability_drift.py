"""The Platform's vendored language table must equal Runtime's (ST-20)."""

import importlib.util
import json
from pathlib import Path

import pytest

PLATFORM_JSON = (
    Path(__file__).resolve().parents[1] / "src" / "platform" / "language_capabilities.json"
)
RUNTIME_MODULE = (
    Path(__file__).resolve().parents[2] / "runtime" / "src" / "runtime" / "language_capabilities.py"
)


def test_platform_table_equals_runtime_table() -> None:
    if not RUNTIME_MODULE.exists():
        pytest.skip("runtime sources are not next to platform")
    spec = importlib.util.spec_from_file_location("_runtime_language_capabilities", RUNTIME_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vendored = json.loads(PLATFORM_JSON.read_text(encoding="utf-8"))
    assert vendored == module.CAPABILITIES, (
        "language_capabilities.json drifted; regenerate with: python "
        "runtime/src/runtime/language_capabilities.py > "
        "platform/src/platform/language_capabilities.json"
    )
    assert json.loads(module.as_json()) == vendored
