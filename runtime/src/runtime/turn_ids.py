"""Per-session ``turn_id`` counter (ST-02a), shared by the event processor, the timeline
tracker and :class:`runtime.turn_debug.TurnDebugObserver`. No Pipecat dependency."""

from __future__ import annotations

import weakref
from typing import Any


class TurnState:
    """The session's turn counter. ``current`` is 0 until the first turn starts."""

    def __init__(self) -> None:
        self.current = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def begin(self) -> int:
        """Start a turn unless one is already open (idempotent); return the current id."""

        if not self._open:
            self.current += 1
            self._open = True
        return self.current

    def end(self) -> None:
        self._open = False

    def new_turn(self) -> int:
        """Always start a new turn (for overlapping speakers, whose turns are not exclusive)."""

        self.current += 1
        return self.current


_states: "weakref.WeakKeyDictionary[Any, TurnState]" = weakref.WeakKeyDictionary()


def turn_state(session: Any) -> TurnState | None:
    """The shared :class:`TurnState` of ``session`` (None without a session)."""

    if session is None:
        return None
    try:
        state = _states.get(session)
        if state is None:
            state = _states[session] = TurnState()
        return state
    except TypeError:  # not weak-referenceable: no sharing, but never an error
        return TurnState()
