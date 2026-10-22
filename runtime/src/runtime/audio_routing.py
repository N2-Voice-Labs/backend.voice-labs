"""Per-listener audio routing for two-person interpreter rooms (RM-001, RM-002).

The audio ownership contract (``docs/BACKEND_ROADMAP.md``)::

    A_original     published by A        -> Runtime only
    B_original     published by B        -> Runtime only
    A_translation  published by Runtime  -> B only    (track ``translation-for-B``)
    B_translation  published by Runtime  -> A only    (track ``translation-for-A``)

plus ``hear_original`` (default off) which lets each person also hear the other's
original microphone. Enforcement is server-side and needs no client cooperation, so it
holds for browsers, SIP phones and API clients alike. :class:`ListenerAudioRouter` does
three things, all re-applied idempotently by :meth:`ListenerAudioRouter.sync`:

1. **Publish one translation track per listener** (``translation-for-<identity>``) with
   the Runtime's own LiveKit connection, for participants that are in the room. Pipecat's
   LiveKit transport publishes a single track and ignores ``transport_destination``; the
   output subclass in ``participant_lanes`` hands every destination frame to
   :meth:`write` instead.
2. **Restrict who may subscribe to the Runtime's tracks**:
   ``LocalParticipant.set_track_subscription_permissions(allow_all_participants=False,
   participant_permissions=[...])`` lists, per listener, only that listener's track sid.
   It is set to "nobody" before the first track is published, so a track is never
   audible to the wrong person, not even for an instant.
3. **Unsubscribe humans from each other's original microphone** with the LiveKit server
   API (``RoomService.UpdateSubscriptions(subscribe=False)``) unless ``hear_original``.
   LiveKit auto-subscribes every participant to every new track, so this is re-applied
   whenever a participant or track appears, in a short burst: the server API call can
   arrive before the auto-subscription it is meant to undo. Measured against a local
   LiveKit server, a participant joining a room still hears the other person's
   microphone for roughly 0.1-0.2 s before the unsubscribe lands (it cannot be closed
   server-side: a token with ``can_subscribe=false`` also blocks the translation track,
   and the server refuses forced subscriptions for such participants).

Everything here is duck-typed over ``livekit.rtc``/``livekit.api`` objects so it is
unit-tested with fakes; only :func:`livekit_api_from_environment` touches the network.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from livekit import rtc

from .participants import ParticipantDirectory, translation_destination

_logger = logging.getLogger(__name__)

# Re-apply schedule after something changed (seconds between consecutive applications).
SYNC_BURST = (0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)

# Server-side participants that may hear originals (recording, other agents).
_TRUSTED_KINDS = tuple(
    value
    for value in (
        getattr(rtc.ParticipantKind, "PARTICIPANT_KIND_EGRESS", None),
        getattr(rtc.ParticipantKind, "PARTICIPANT_KIND_AGENT", None),
    )
    if value is not None
)


class AudioRoutingError(RuntimeError):
    """Per-listener routing cannot be enforced (e.g. no LiveKit API credentials)."""


@dataclass(frozen=True)
class LiveKitApiSettings:
    url: str
    api_key: str
    api_secret: str

    @property
    def http_url(self) -> str:
        if self.url.startswith("wss://"):
            return "https://" + self.url[len("wss://") :]
        if self.url.startswith("ws://"):
            return "http://" + self.url[len("ws://") :]
        return self.url

    @classmethod
    def from_environment(cls, url: str | None = None) -> "LiveKitApiSettings":
        url = url or os.getenv("LIVEKIT_URL", "")
        key = os.getenv("LIVEKIT_API_KEY", "")
        secret = os.getenv("LIVEKIT_API_SECRET", "")
        if not (url and key and secret):
            raise AudioRoutingError(
                "two-person interpreter sessions need LIVEKIT_URL, LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET to enforce per-listener audio"
            )
        return cls(url, key, secret)


def livekit_api_from_environment(settings: LiveKitApiSettings) -> Any:
    from livekit import api

    return api.LiveKitAPI(settings.http_url, settings.api_key, settings.api_secret)


@dataclass
class _ListenerTrack:
    identity: str
    source: Any
    track: Any
    sid: str
    name: str


class ListenerAudioRouter:
    """Owns the per-listener tracks and the subscription policy of one interpreter room."""

    def __init__(
        self,
        directory: ParticipantDirectory,
        *,
        hear_original: bool = False,
        api_settings: LiveKitApiSettings | None = None,
        api_factory: Callable[[LiveKitApiSettings], Any] = livekit_api_from_environment,
        burst: Sequence[float] = SYNC_BURST,
    ) -> None:
        self._directory = directory
        self.hear_original = hear_original
        self._api_settings = api_settings
        self._api_factory = api_factory
        self._api: Any | None = None
        self._burst = tuple(burst) or (0.0,)
        self._room: Any | None = None
        self._audio_format = (24000, 1, 1000)
        self._tracks: dict[str, _ListenerTrack] = {}
        self._applied_permissions: dict[str, tuple[str, ...]] | None = None
        self._tracks_lock = asyncio.Lock()
        self._originals_lock = asyncio.Lock()
        self._runner: asyncio.Task[None] | None = None
        self._rerun = False
        self._wake = asyncio.Event()
        self._closed = False
        self._subscribed_sids: set[tuple[str, str]] = set()
        directory.add_listener(self._on_participant)

    # -- wiring -----------------------------------------------------------------

    async def attach(
        self,
        room: Any,
        *,
        sample_rate: int,
        num_channels: int = 1,
        queue_size_ms: int = 1000,
    ) -> None:
        """Start enforcing on a connected room (the Runtime has joined it)."""

        self._room = room
        self._audio_format = (sample_rate, num_channels, queue_size_ms)
        for event in (
            "participant_connected",
            "participant_disconnected",
            "track_published",
            "track_unpublished",
        ):
            room.on(event, self._on_room_event)
        # Close everything first: no one may subscribe to a Runtime track until a policy
        # names its audience. Then reconcile with whoever is already in the room.
        self._set_permissions({})
        await self.sync()
        self.request_sync()

    def _on_room_event(self, *_: object) -> None:
        self.request_sync()

    async def _on_participant(self, _participant: object) -> None:
        self.request_sync()

    def request_sync(self) -> None:
        """Re-apply the policy now and again shortly after (coalesced)."""

        if self._closed or self._room is None:
            return
        if self._runner is not None and not self._runner.done():
            # Restart the burst now instead of after its current pause.
            self._rerun = True
            self._wake.set()
            return
        self._runner = asyncio.get_running_loop().create_task(self._run())

    async def _pause(self, delay: float) -> None:
        """Sleep, but wake early when something changed (``request_sync``)."""

        if self._rerun:
            return
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), delay)
        except asyncio.TimeoutError:
            pass

    async def _run(self) -> None:
        reconcile: asyncio.Task[None] | None = None
        try:
            while not self._closed:
                self._rerun = False
                for delay in self._burst:
                    if delay:
                        await self._pause(delay)
                        if self._rerun:
                            break
                    snapshot = self._snapshot()
                    if snapshot is None:
                        return
                    room, present = snapshot
                    # Publishing is a slow server round trip; never let it delay the
                    # (time-critical) unsubscriptions of the next step.
                    if reconcile is None or reconcile.done():
                        if reconcile is not None:
                            reconcile.result()
                        reconcile = asyncio.get_running_loop().create_task(
                            self._reconcile_tracks(room, present)
                        )
                    await self._enforce_originals(room)
                    if self._rerun:
                        break
                if not self._rerun:
                    break
            if reconcile is not None:
                await reconcile
        except asyncio.CancelledError:
            if reconcile is not None and not reconcile.done():
                reconcile.cancel()
            raise
        except Exception:
            _logger.exception("Listener audio synchronisation failed")

    # -- the policy ---------------------------------------------------------------

    async def sync(self) -> None:
        """Reconcile tracks, track permissions and subscriptions with the room.

        Hiding the originals is time-critical (a listener hears them until it lands) while
        publishing a track takes a server round trip, so the two run concurrently.
        """

        snapshot = self._snapshot()
        if snapshot is None:
            return
        room, present = snapshot
        await asyncio.gather(self._enforce_originals(room), self._reconcile_tracks(room, present))

    def _snapshot(self) -> tuple[Any, set[str]] | None:
        """The room and who is in it; also tells the directory who is present."""

        room = self._room
        if room is None or self._closed:
            return None
        present = set(getattr(room, "remote_participants", {}) or {})
        self._directory.set_present(present & set(self._directory_identities()))
        return room, present

    async def _reconcile_tracks(self, room: Any, present: set[str]) -> None:
        async with self._tracks_lock:
            await self._sync_tracks(room, present)
            self._set_permissions(
                {identity: (entry.sid,) for identity, entry in self._tracks.items()}
            )
            await self._subscribe_listeners(room)

    def _directory_identities(self) -> list[str]:
        return [context.identity for context in self._directory.contexts()]

    async def _sync_tracks(self, room: Any, present: set[str]) -> None:
        wanted = [identity for identity in self._directory_identities() if identity in present]
        for identity in list(self._tracks):
            if identity not in wanted:
                await self._unpublish(room, identity)
        for identity in wanted:
            if identity not in self._tracks:
                await self._publish(room, identity)

    async def _publish(self, room: Any, identity: str) -> None:
        name = self._directory.context(identity).output_track
        sample_rate, channels, queue_ms = self._audio_format
        source = rtc.AudioSource(sample_rate, channels, queue_size_ms=queue_ms)
        track = rtc.LocalAudioTrack.create_audio_track(name, source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        try:
            publication = await room.local_participant.publish_track(track, options)
        except Exception:
            await _close_quietly(source)
            raise
        self._tracks[identity] = _ListenerTrack(identity, source, track, publication.sid, name)
        _logger.info("Published %s (%s) for listener", name, publication.sid)

    async def _unpublish(self, room: Any, identity: str) -> None:
        entry = self._tracks.pop(identity, None)
        if entry is None:
            return
        self._subscribed_sids = {pair for pair in self._subscribed_sids if pair[1] != entry.sid}
        try:
            await room.local_participant.unpublish_track(entry.sid)
        except Exception:
            _logger.warning("Could not unpublish %s", entry.name, exc_info=True)
        await _close_quietly(entry.source)

    def _set_permissions(self, audience: dict[str, tuple[str, ...]]) -> None:
        """Only each listener may subscribe to the Runtime track(s) listed for them."""

        if audience == self._applied_permissions or self._room is None:
            return
        permissions = [
            rtc.ParticipantTrackPermission(
                participant_identity=identity,
                allow_all=False,
                allowed_track_sids=list(sids),
            )
            for identity, sids in audience.items()
        ]
        self._room.local_participant.set_track_subscription_permissions(
            allow_all_participants=False, participant_permissions=permissions
        )
        self._applied_permissions = dict(audience)

    async def _enforce_originals(self, room: Any) -> None:
        """Unsubscribe everyone from the microphones of the two people (unless allowed)."""

        async with self._originals_lock:
            directory_ids = set(self._directory_identities())
            remote = dict(getattr(room, "remote_participants", {}) or {})
            originals: dict[str, list[str]] = {}
            for identity in directory_ids:
                participant = remote.get(identity)
                if participant is not None:
                    originals[identity] = _audio_sids(participant)
            calls = []
            for viewer in remote.values():
                if getattr(viewer, "kind", None) in _TRUSTED_KINDS:
                    continue
                forbidden: list[str] = []
                for owner, sids in originals.items():
                    if owner == viewer.identity:
                        continue
                    if self.hear_original and viewer.identity in directory_ids:
                        continue
                    forbidden.extend(sids)
                if forbidden:
                    calls.append(self._update_subscriptions(room, viewer.identity, forbidden, False))
            if calls:
                await asyncio.gather(*calls)

    async def _subscribe_listeners(self, room: Any) -> None:
        """Clients that do not auto-subscribe still get their own translation track."""

        remote = dict(getattr(room, "remote_participants", {}) or {})
        for identity, entry in list(self._tracks.items()):
            if identity in remote and (identity, entry.sid) not in self._subscribed_sids:
                if await self._update_subscriptions(room, identity, [entry.sid], True):
                    self._subscribed_sids.add((identity, entry.sid))

    async def _update_subscriptions(
        self, room: Any, identity: str, sids: list[str], subscribe: bool
    ) -> bool:
        try:
            client = await self._api_client()
            from livekit import api

            await client.room.update_subscriptions(
                api.UpdateSubscriptionsRequest(
                    room=room.name, identity=identity, track_sids=sids, subscribe=subscribe
                )
            )
            return True
        except Exception:
            _logger.warning(
                "LiveKit update_subscriptions(%s, subscribe=%s) failed", identity, subscribe,
                exc_info=True,
            )
            return False

    async def _api_client(self) -> Any:
        if self._api is None:
            settings = self._api_settings or LiveKitApiSettings.from_environment()
            result = self._api_factory(settings)
            self._api = await result if isinstance(result, Awaitable) else result
        return self._api

    # -- audio output ----------------------------------------------------------------

    def destination_for(self, identity: str) -> str | None:
        lane = self._directory.lane_of(identity)
        return translation_destination(lane) if lane is not None else None

    async def write(self, destination: str, frame: Any) -> bool:
        """Publish a frame on the track belonging to ``destination`` (a listener lane)."""

        lane = _lane_of_destination(destination)
        identity = self._directory.identity_for_lane(lane) if lane is not None else None
        entry = self._tracks.get(identity) if identity else None
        if entry is None:
            return True  # that listener is not in the room: nothing to play to
        try:
            await entry.source.capture_frame(frame)
            return True
        except Exception as exc:
            if "InvalidState" not in str(exc):
                _logger.error("Error publishing translation audio: %s", exc)
            return False

    def clear(self) -> None:
        for entry in self._tracks.values():
            clear = getattr(entry.source, "clear_queue", None)
            if clear is not None:
                clear()

    def published_tracks(self) -> dict[str, str]:
        """``identity -> track sid`` of the translation tracks currently published."""

        return {identity: entry.sid for identity, entry in self._tracks.items()}

    async def close(self) -> None:
        self._closed = True
        runner, self._runner = self._runner, None
        if runner is not None and not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        async with self._tracks_lock:
            for entry in list(self._tracks.values()):
                await _close_quietly(entry.source)
            self._tracks.clear()
            api_client, self._api = self._api, None
        if api_client is not None:
            await _close_quietly(api_client)


def _audio_sids(participant: Any) -> list[str]:
    return [
        publication.sid
        for publication in participant.track_publications.values()
        if publication.kind == rtc.TrackKind.KIND_AUDIO
    ]


def _lane_of_destination(destination: str) -> int | None:
    try:
        lane = int(destination.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None
    return lane if lane >= 0 else None


async def _close_quietly(resource: Any) -> None:
    for name in ("aclose", "close"):
        closer = getattr(resource, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if isinstance(result, Awaitable):
                await result
        except Exception:
            _logger.debug("Closing %r failed", resource, exc_info=True)
        return
