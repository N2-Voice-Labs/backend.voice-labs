# Translated calls by shareable link

Two people who speak different languages open a web page, one sends the other a link, and they talk
through the existing Runtime (ElevenLabs STT/TTS, Google Cloud Translation with Gemini fallback, LiveKit).
No app, no account. This is a test harness for the product, not a distribution channel.

Status: implemented and tested against a local Platform, LiveKit Cloud and a Chromium/Chrome browser
(desktop and phone-sized windows). **Not yet verified:** a real spoken Korean-English call, a real phone,
Safari/iOS, Android. Section 7 lists what to test by hand.

## 1. What a person does

**Person A** opens `/call`, picks "I speak" and "They speak", checks the microphone, presses **Start call**,
presses **Create invite link**, and sends the link by any messenger.
**Person B** opens the link, sees who invited them and the two languages, checks the microphone, presses
**Join call**. Either person presses **End call** to hang up (it ends for both). Mute is the round
microphone button.

"Call options" on the first screen has the pace (Fast / Balanced / Careful) and **Audio**: *Translation only*
(default) or *Also their real voice, quietly (test)* (cinematic dubbing, see [dubbing](dubbing.md), the original
ducks to -24 dB while the translation speaks). The choice is the creator's and applies to both people.

## 2. How it works

```
browser A ── POST /calls ───────────────► Platform ──► hidden interpreter session (same as /translations)
          ◄─ LiveKit token (10 min) + session token (this call only)
browser A ── POST /calls/{id}/invite ───► random code, only its SHA-256 is stored, expires in 30 min
browser B ── GET  /calls/invites/{code} ► who invited you, your language (seats nobody)
browser B ── POST /calls/invites/{code}/join ► seats the guest: LiveKit token + session token
both      ── LiveKit room (audio) ──────► Runtime interpreter (STT → translate → TTS → each listener's own track)
either    ── POST /calls/{id}/end ──────► call ends for both; Runtime and rows cleaned up
```

Nothing about translation changed: `POST /calls` calls the same code as `POST /translations`, and joining uses the same
participant path as `POST /voice/sessions/{id}/participants`. The Runtime, the audio routing and the latency path are untouched. The link carries only the invitation code, in the URL `#fragment`, which browsers never send to a server;
`/join` strips it from the address bar at once.

`platform/src/platform/calls.py` is the whole backend feature; the frontend is the sibling `frontend` repository
(`lib/platform.ts` call functions, `components/features/call/*`, `lib/invite.ts`).

## 3. Start it locally

Needs: the LiveKit server (or LiveKit Cloud keys), Python 3.11+, Node 20.9+.

```bash
# 1. LiveKit dev server + database (optional: SQLite works for a test), from stack.voice-labs
./compose.sh up -d        # LiveKit on :7880 (devkey/secret); or use LiveKit Cloud URL and keys

# 2. Platform
cd backend.voice-labs/platform
export PLATFORM_DATABASE_URL=sqlite:///./calls.db PLATFORM_AUTO_CREATE_SCHEMA=1
export PLATFORM_PUBLIC_CALLS=1                       # the feature is OFF unless this is set
export PLATFORM_API_KEYS=$(openssl rand -hex 32)     # keep authentication ON: the browser needs no key
export PLATFORM_WEB_ORIGIN=http://localhost:3000     # the frontend's origin (CORS)
export PLATFORM_RUNTIME_TOKEN=local-runtime-token RUNTIME_API_URL=http://localhost:8001
export LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
export PLATFORM_CALLS_PROVIDERS="stt=elevenlabs,llm=google-translate,tts=elevenlabs"
uvicorn src.platform.main:app --port 8000

# 3. Runtime (the same LiveKit and token; the provider keys live only here)
cd backend.voice-labs/runtime
export RUNTIME_PLATFORM_URL=http://localhost:8000 RUNTIME_PLATFORM_TOKEN=local-runtime-token
export LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
export ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=... GOOGLE_TRANSLATE_API_KEY=... GEMINI_API_KEY=...
PYTHONPATH=src uvicorn runtime.api:app --port 8001

# 4. Frontend
cd frontend
echo 'NEXT_PUBLIC_PLATFORM_URL=http://localhost:8000' > .env.local
npm ci && npm run dev      # http://localhost:3000/call
```

The keys in `backend.voice-labs/.env` can be loaded with `set -a; . ./.env; set +a`. Never put them in the frontend.

**Two devices.** A phone cannot reach your `localhost`, and browsers only allow the microphone on `https://` pages
(or `localhost`). Put the frontend and the Platform behind HTTPS (a tunnel such as `cloudflared tunnel --url http://localhost:3000`
and another for port 8000, or a deployment), set `NEXT_PUBLIC_PLATFORM_URL` to the Platform's HTTPS URL, and add the
frontend's HTTPS origin to `PLATFORM_WEB_ORIGIN`. LiveKit Cloud is already `wss://`. A local `ws://localhost:7880` LiveKit
server is unreachable from a phone; use LiveKit Cloud or expose it. Behind a proxy you run, set `PLATFORM_TRUST_PROXY=1`
so the per-address limits see the real client address.

## 4. Configuration

| Variable (Platform) | Default | Meaning |
|---|---|---|
| `PLATFORM_PUBLIC_CALLS` | off | Turns the three public routes on. While off they answer 404. |
| `PLATFORM_CALLS_PROVIDERS` | none (Runtime defaults) | `stt=…,llm=…,tts=…` Runtime provider names every call uses. Provider records are created on first use. The browser cannot choose them. |
| `PLATFORM_CALLS_ORG_ID` | default organisation | Organisation the public calls belong to (its session cap applies). |
| `PLATFORM_CALL_INVITE_TTL_SECS` | 1800 | How long an invitation code works (minimum 30). |
| `PLATFORM_CALLS_PER_IP_HOUR` | 6 | Calls a client address may start per hour (0 turns the limit off). |
| `PLATFORM_CALL_OPEN_PER_IP_MINUTE` | 30 | Invitation lookups and joins per client address per minute. |
| `PLATFORM_TRUST_PROXY` | off | Take the client address from `X-Forwarded-For` (only behind your own proxy). |
| `PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG` | 10 | Simultaneous calls; more get 429 `too_many_active_sessions`. |
| `PLATFORM_STALE_SESSION_SECS` | 300 | A call with no activity this long is closed by the reaper (a call nobody joined). |
| `PLATFORM_API_KEYS` | none | **Set it on any server others can reach.** The three public routes work without it; everything else then requires a key. |

Also required (as for any call): LiveKit URL/key/secret on both services, the shared Runtime token, and the Runtime's provider keys.
Frontend: `NEXT_PUBLIC_PLATFORM_URL`.

## 5. Security and abuse limits

| Requirement | How |
|---|---|
| Unpredictable links | 128-bit random secret; only its SHA-256 digest is stored; compared in constant time; wrong, malformed and foreign codes all answer 404. |
| Expiring invitations | Default 30 minutes (`PLATFORM_CALL_INVITE_TTL_SECS`); 410 `invite_expired`; a new link replaces the old. |
| Short-lived LiveKit tokens | 10 minutes, issued only when a person starts or joins; the invitation itself holds none. |
| Two people at most | The second seat is created when the guest joins; any other code or device gets 409 `guest_already_invited`; `session_full` beyond two. Re-opening the same link is the same guest (a reconnect or a second device: it takes over the seat, it does not add a third person). |
| No credentials in browsers | The browser holds a session token (`vls_…`) scoped to one call, usable only for that call's routes (read transcript, invite for the creator, hang up). Provider and Platform admin keys stay on the servers. |
| Unauthorised joins | The code is the only way in; the guest's token cannot create invitations, calls or read anything else; tested with authentication on. |
| Cleanup | Hang-up from either side ends the session, the Runtime session and the room; a call nobody joined is closed by the reaper (`PLATFORM_STALE_SESSION_SECS`); finished calls refuse new joins (410 `call_ended`). |
| Resource abuse | Per-address limits on creating calls and on opening invitations (429 with `Retry-After`), the per-organisation cap on simultaneous calls, field-length limits. |

**Known weak points** (prototype): anyone who can reach the server can start up to the limits (there is no sign-in), which spends
provider credits; if that matters, keep the URL private, lower the limits and the cap, or put the site behind an access gate
(Cloudflare Access, basic auth). The invite link is a bearer secret: whoever opens it first, or opens it again, gets the guest seat.

## 6. What was tested automatically

- Platform: `platform/tests/test_calls.py` (18 tests: off by default; creation; cinematic selection; provider selection; invite, info, join;
  reopen = same guest, a different code is refused; secret not stored; replacement, malformed, foreign and expired codes; ended calls;
  hang-up by either person; rate limits; session cap; with authentication on, the public routes need no credentials while admin routes stay closed
  and a session token only reaches its own call) and the cross-organisation isolation tests (`test_tenancy_isolation.py`) updated for the new routes.
- Frontend: vitest (`__tests__/call-experience`, `join-invite`, `platform-call`): start, invite, join, error states, the cinematic switch, request
  shapes without credentials, the code never rendered or logged.
- Real services, scripted browsers (`scripts/calls_flow_check.cjs`, run it against a running stack; 10 checks passed): a local Platform with authentication on, LiveKit Cloud, and headless Chrome with a fake
  microphone: create → invite → join (desktop creator, phone-sized and iPhone-emulated guest) → both in the same room → mute → hang-up by the
  guest and by the creator → the other side shows "ended" → the link then says the call has ended → the same link on a second device is the same seat →
  no code or token in the guest's URL or storage → no horizontal overflow on a phone. Runtime providers were stubs: **no translation ran in
  these checks**.

## 6a. First real-provider run (2026-11-07, two phones, ElevenLabs STT/TTS + Google Translate, LiveKit Cloud)

Two phones joined through a public tunnel (an SSH relay, closed afterwards). What happened, from the Platform and Runtime logs:

| Call | Result |
|---|---|
| 1 | **Worked end to end.** English speech was transcribed and spoken back in Korean for the other phone: "You are correct. Thank you for that." → "맞습니다. 알려주셔서 감사합니다." and 3 more exchanges. Ended by the Runtime's idle timeout (nobody pressed End). |
| 2 | Both phones in; the speakers talked **Russian** while the call was set to English ↔ Korean. Each person's speech-to-text is fixed to the language chosen for them, so the transcripts were garbled ("А сейчас еще, еще одно link…") and the translations followed. Not a bug, but the page should say that you must speak the language you picked. A few correct short exchanges ("Hello! Hello, hello!" → "안녕하세요!…"). Ended with a provider failure (below). |
| 3 | **ElevenLabs "You have exceeded your quota."** The free account (10,000 characters) was used up. The Runtime reported `provider_failed` after about 16 s, and every later `POST /calls` was refused with 503 `provider_unavailable` (the Platform checks the providers first and refuses to start a call it cannot serve; this is the intended fail-closed behavior). The page now explains this in plain words. The quota resets around 2026-12-05, or upgrade the plan. |

Measured on the Runtime (`RUNTIME_TIMELINE_DIAG=1`, two turns that completed): speech end → final transcript 418–425 ms, → turn end ≈ 500 ms, → first translation token 599–731 ms, → first translated audio frame 974–1180 ms Runtime-side; one turn reached LiveKit later (3053 ms, 781 ms in the submit step) and is not explained. STT first-result 0.41–0.50 s; TTS first byte 0.17–0.42 s. Only two turns carried the full timeline, so this is an indication, not a distribution. The ElevenLabs account counted 10,000 of 10,000 characters; the STT log summed to about 193 s of audio across the three calls. Translation accuracy, echo, phone-specific behavior and the cinematic option were **not** assessed in this run.

## 6b. Audio path: the browser must join the private listen room (fixed 2026-11-08)

With private listening (the default for two-person calls) the session-room LiveKit token **cannot subscribe to anything**, and each
person's translation is published by the Runtime on `translation-for-<identity>` in a separate subscribe-only room
(`connection.listen`, see [contracts](contracts.md)). The call page used to connect only to the session room, so **a browser never
received any translated audio** (the transcript worked, the sound did not). `useCallRoom` now joins the listen room next to the session
room (`CallCredentials.listen`, passed from the create and join replies) and plays that track through the same element path as before. The earlier
two-phone call's transcript is therefore real, but whether anybody heard the voice is unknown; it has to be re-run with the fixed page.

## 6c. Browser-side audio latency (opt-in)

Off by default. Switch on with `?latency=1` in the page address (on `/call`, or `/join?latency=1#c=…`) or `localStorage["vl-latency-diag"] = "1"`;
read it with `window.__voiceLabsLatency.export()` in the console. It taps the **translated track this person listens to** (never their own
microphone), adds no node to the playback path, and records only event names, `performance.now()` times, identities, track names and counts: no speech and no credentials.

| Event | Exact meaning |
|---|---|
| `listener_room_connected` | the listen room's connect resolved |
| `translated_track_subscribed` | the `translation-for-<identity>` audio track was subscribed (tagged with `listener`, `speaker` = the other participant seen in the session room, `track`) |
| `first_rtp_audio_received` | the receiver's `bytesReceived` first became > 0 (polled every 250 ms; says the track carries audio, not that a translation started) |
| `browser_first_voiced_audio` (`seq` n) | the page's analyser first read non-silent PCM (RMS ≥ 0.01 for ≥ 40 ms) from the track for the nth voiced stretch, backdated to its first voiced sample; resolution 20 ms, later in a background tab. **Not confirmed audibility**: the output device, volume and the OS are out of view |
| `browser_voiced_audio_ended` (`seq` n) | the last voiced sample before 400 ms of silence |
| `playback_started` / `playback_failed` / `playback_blocked` / `playback_allowed` | the element's `playing`, a rejected `play()` (reason = error name, e.g. `NotAllowedError`), the page's autoplay status |
| `element_waiting` / `element_stalled` / `element_error`, `audio_concealment` | buffering, stalls, decode errors; concealed samples reported by the receiver (loss or underrun) |
| `listener_reconnecting` / `listener_reconnected` / `listener_disconnected`, `listener_connect_failed`, `analyser_unavailable`, `receiver_stats_unavailable` | connection trouble, and measurements that could not be made (with the reason) |

**Limits.** Times are this browser's clock and are never subtracted from a Runtime timestamp (another machine, another clock): compare durations
(the gap between onsets, connection to first onset). Nothing in the audio carries a `turn_id`, so utterances are identified by their ordinal on each
listener's track, which holds only while no utterance is dropped, merged or split; pairing them with the Runtime's `utterance_timeline` events
(`speaker`, `listener`, `destination`) is by order per listener. A tab in the background throttles timers: its onsets are late.

**Validation without providers or credits** (`scripts/browser_latency_check.cjs` + `scripts/synthetic_translation_publisher.py`, needs a running Platform without a Runtime,
the built frontend and LiveKit): a synthetic publisher plays a short translation twice (0.5 s, 3 s apart) on the creator's listen track and one long one (3 s) on the guest's,
starting together; real Chrome pages record. Result, 2026-11-08: all checks passed. Onset numbering per listener, silence not counted, the creator's first audio 311 ms after hand-over
and the guest's 322 ms (the 3 s translation did not delay the other listener: 12 ms apart), utterance lengths 0.5 s and 3.00 s, the 3 s gap read as 3.02 s, no failures. These figures include the
publisher's ~200 ms lead over real time and a loopback host clock: they validate the instrument, they are not a Runtime latency.

## 7. What needs manual testing on real devices (do not treat any of this as verified)

1. A spoken Korean ↔ English call, desktop and phone: both hear the right translation, in the right ears, no echo of their own voice.
2. iOS Safari and Android Chrome: microphone permission prompt, audio playback after the first tap, backgrounding the tab, screen lock, switching Wi-Fi/cellular.
3. Reconnection: kill the network for 10-20 s on each side and watch both pages recover or say so.
4. Headphones vs speakers: the other person's translation leaking into your microphone (also with the cinematic option).
5. AI-only vs cinematic (the Audio option): intelligibility, whether the original voice helps or distracts; this is the listening study described in
   [Dubbing](dubbing.md), on a real call.
6. Latency end to end: use `RUNTIME_TIMELINE_DIAG=1` (the timeline events are described in [Contracts](contracts.md)); the link adds nothing to the translation path.
7. Interruptions, overlapping speech, short utterances, sentence splitting, translation accuracy.
8. A link sent through KakaoTalk/WhatsApp/SMS: the in-app browser may not allow the microphone; the page should be opened in the system browser (not verified).
9. Abuse: the per-address limits behind your real proxy (`PLATFORM_TRUST_PROXY`).

## 8. Limitations

- The call needs the Runtime and real provider keys to translate; without them the pages work but nobody is translated ("Interpreter starting…").
- A guest who closes the tab and reloads loses the page state; re-opening the link from the messenger gets the same seat back (within 30 minutes of the link's creation).
- Nobody can *leave* without ending the call for both. Two people only, no waiting room, no names beyond what they type, no transcript download.
- The first screen reads the voice library (`/voices`), which needs an admin key, so with authentication on the "My translated voice" choice never appears (the call uses the default voice).
- `frontend/__tests__/call-poll.test.tsx` has one failing test that also fails without this work (it depends on the current date).
