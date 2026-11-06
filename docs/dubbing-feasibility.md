# Keeping the original voice under the translation: feasibility study

Status: study only, nothing implemented. Question (from `docs/goal.md`): can each listener hear
the speaker's real voice quietly under the translated voice (a "cinematic dubbing" feel), without
adding translation latency, inside the current LiveKit design?

"Measured" means measured in this repository with free probes; "hypothesis" means not yet tested.

## What exists today

- The Runtime receives every participant's microphone, resamples it per participant (16 kHz) and
  feeds each lane's VAD and STT (`participant_lanes.py`).
- It publishes one translation track per listener (`translation-for-<identity>`, 24 kHz mono,
  40 ms chunks, `rtc.AudioSource` with a 1000 ms queue), optionally in a private listen room
  (`audio_routing.py`). Humans are unsubscribed from each other's microphones.
- `hear_original` (default off) lets each person also receive the other's raw microphone track.
  It is plain track forwarding by the SFU: no ducking, no delay, equal loudness, mixing left to the
  client. With it on, the private listen room and the phone room are not used (`pipeline.py`), so
  the isolation guarantees of `audio_routing.py` are given up.
- The audio path the original takes today with `hear_original`: speaker, SFU, listener directly
  (about 100 ms, measured one-way floor 100-110 ms). The translation arrives about 1.26 s after the
  speaker stops (measured, Balanced profile).

## Options

### A. Client-side mix (`hear_original=true`, client does gain, ducking and delay)

- Translation latency: unchanged. Runtime work: none.
- Needs the web client to lower the original (`track volume`), duck it when the translation track
  is active (analyser on that track) and optionally delay it (`DelayNode`). The client lives in a
  sibling repository (not touched here).
- Phones (SIP legs) cannot be controlled: they would hear both voices at equal level.
- Gives up private listening (see above) and depends on every client doing it right.
- Echo: a listener without headphones plays the speaker's original from speakers into their own
  microphone, which feeds their STT lane. Browser echo cancellation covers audio played in the
  page; a phone handset does not. Hypothesis, not tested.

### B. Runtime-side mix into the per-listener translation track (recommended)

The Runtime already holds both ingredients: the speaker's original audio (lane input) and the
listener's output track. A small mixer per listener would:

1. keep a short delay line per speaker (original, 16 kHz, resampled to the track's 24 kHz);
2. receive the TTS frames for that listener;
3. every 20 ms, output `original * g_orig + translation`, where `g_orig` is a low base gain while
   only the speaker is talking and ducks further, with a fast attack and a slow release, while the
   translation plays;
4. write the result to the same `AudioSource` the translation uses today.

Properties:

- **Works for every client**, including phones, and the listener never receives the speaker's
  microphone track, so `hear_original` stays off and private listening / unsubscribe protections
  are unchanged. This is the main advantage over A.
- **The original is already late by about 310 ms**, with no artificial delay: speaker to Runtime
  (measured uplink 161 ms) plus Runtime to listener (measured downlink 149 ms), against about
  100 ms on the SFU's direct path. That is the "slightly delayed original" asked for. Extra delay
  would only prolong the overlap with the translation start and is not needed (hypothesis: a
  listening test decides; default 0).
- **Translation latency**: STT, translation and TTS are untouched. The only added step is the output
  clock: a TTS frame may wait for the next 20 ms tick, so at most one tick (20 ms, design bound; to be
  measured with `scripts/transport_probe.py`, free). The mixer should write at most a few tens of
  milliseconds ahead of real time, so the ducking follows what is actually playing; today's
  burst writes into a 1000 ms queue would make the duck start early or end late.
- **Cost, measured**: mixing 20 ms of audio takes about 15 microseconds (0.08% of one core per
  listener, naive resampling; a real resampler is two to three times that). A 300 ms delay line is
  9.6 kB per speaker. The load is negligible next to STT and TTS.
- **Continuous track**: the mixer publishes a steady stream (silence between sentences). Opus DTX
  keeps silence cheap; to be confirmed. Today the track is idle between translations, which showed
  a 150-180 ms first-frame penalty after 8 s idle in one direction (measured, unexplained). A steady
  stream might remove it (hypothesis) and is worth measuring.
- **Complexity**: one new module (about 200-300 lines) plus a per-session setting, default off,
  and fakes-based tests like `audio_routing`. No new dependency.

### C. Hybrid (Runtime publishes a second, processed "original" track, the client mixes)

More tracks and more client work than B, no phone support, and the ducking signal would need to
travel to the client. Not recommended.

## Technical challenges and risks, in order

1. **Echo into the listener's own STT lane** (both options). A louder original in a speakerphone
   call can be transcribed as the listener's speech. Mitigations: low base gain, browser echo
   cancellation, the existing lane gate, and gating the listener's lane while the mixed output is
   loud. Needs a test with recorded audio before any rollout.
2. **Ducking quality**: attack and release times, base gain and duck depth are perceptual. Starting
   points (hypothesis): original about 15 dB under the translation while only the original is
   playing, about 20-25 dB while the translation speaks, 30 ms attack, 300 ms release.
3. **Voice-activity of the original**: ducking can follow the translation alone (simple, no extra
   detector); fade the original out when the speaker stops, using the VAD the lane already runs.
4. **Clock and drift**: the mixer needs one steady clock per listener track and must handle the
   speaker leaving, a second speaker interrupting, and a listener joining late.
5. **Privacy and policy**: letting B hear A's real voice is what `private_listening` was designed
   to prevent. It must be an explicit session option (and a visible one to A), default off.
6. **Video**: the original is about 310 ms late relative to a video of the speaker. Irrelevant for
   audio-only calls; for video calls the delay would have to be reduced or the video delayed.
   Not known whether video is planned.
7. **Voice cloning is a separate lever**: translations can already be spoken in the speaker's own
   voice (`voice` per participant). Mixing the real voice and a cloned translation voice of the same
   speaker can sound odd; to judge in the listening test.

## Recommendation

Build B, behind a session setting that defaults to off, in three free steps before any live call:

1. A mixer module with fakes-based unit tests (gain, ducking envelope, delay line, speaker leaving).
2. The free transport probe extended to run the mixer, to measure the added latency per frame
   (expected 0-20 ms) and the effect of a steady stream.
3. An offline listening test on recorded samples (`runtime/bench/samples`) with a few base gain and
   duck settings, then a two-person trial with headphones and with speakers to check the echo.

Interim, without Runtime work: try `hear_original=true` in a lab room to hear what the raw effect
is like (no ducking), keeping in mind that it drops private listening.

Estimated effect on latency: none for the translation path; at most one 20 ms output tick,
to be measured. Effect on trust and naturalness: a hypothesis until the listening test.
