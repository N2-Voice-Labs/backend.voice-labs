import importlib.util
import sys
from pathlib import Path

import pytest

from src.platform.runtime_events import ACCEPTED_EVENT_TYPES, RUNTIME_STATE_EVENT_TYPES

RUNTIME_EVENTS_PATH = (
    Path(__file__).resolve().parents[2] / "runtime" / "src" / "runtime" / "events.py"
)


def load_runtime_event_types() -> tuple[str, ...]:
    # Loaded from file so the runtime package is not imported into the platform process.
    name = "_runtime_events_for_drift_test"
    spec = importlib.util.spec_from_file_location(name, RUNTIME_EVENTS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module.EVENT_TYPES


@pytest.mark.skipif(
    not RUNTIME_EVENTS_PATH.is_file(), reason="sibling runtime repository not present"
)
def test_accepted_event_types_match_runtime_event_types():
    assert set(load_runtime_event_types()) == ACCEPTED_EVENT_TYPES


def test_runtime_state_event_types_are_accepted_subset():
    assert RUNTIME_STATE_EVENT_TYPES <= ACCEPTED_EVENT_TYPES
