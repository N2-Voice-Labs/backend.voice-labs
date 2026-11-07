# Cinematic dubbing (prototype, off by default)

The listener normally hears only the translated voice. With dubbing on, they also hear the other
person's real voice, quiet and delayed a little, ducked further while the translation speaks, so a
real human stays present under the translator. Whether this makes calls feel more trustworthy is a
product hypothesis that still needs listeners; nothing here proves it.

Status: working prototype, tested offline and through LiveKit with tones; **not yet heard by people**.
Design background and rejected options: [dubbing-feasibility.md](dubbing-feasibility.md).

## Using it

- Platform: `POST /translations` with `"dubbing": true` (or `"subtle"`, `"balanced"`, `"present"`).
  Absent or `false` is exactly today's behaviour: the setting is not even stored in the hidden agent's
  config, so existing agents and their hashes are unchanged. It cannot be combined with `hear_original`.
- Runtime: `interpreter.dubbing` takes the same value, or an object
  `{"profile": "balanced", "delay_ms": 150, "base_gain_db": -10, "duck_gain_db": -20, "attack_ms": 120,
  "release_ms": 500, "gate_db": -50, "gate_hold_ms": 200}` (every number optional).
- While a call runs, `ListenerAudioRouter.set_dubbing(profile | None)` switches it on, off or to another
  profile without rebuilding the pipeline. Switching off sends any translated audio still in the mix
  straight to the track, so a sentence is not cut. (Nothing in the public API calls it yet; it is the hook
  for a future per-call control.)

## How it works

```
speaker's mic -> per-participant resampler -> lane (VAD, STT, turns, translation, TTS) ... unchanged
                                   \-> copy -> ListenerAudioRouter.feed_original -> other listener's DubMixer
TTS frames for a listener ---------------> that listener's DubMixer (instead of capture_frame)
DubMixer -> (every 10 ms, only while there is sound) -> the listener's AudioSource (the same track as today)
```

- **The translation is never delayed or reordered.** A TTS frame goes into the mix on the next block.
  Idle that is immediate; while the original is playing it waits for at most the audio already queued
  (about one 10 ms block). The original adapts to the translation, not the other way round.
- **Delay**: the original is held `delay_ms` (default 150, minimum 40 for jitter) on top of the natural
  path through the Runtime (about 310 ms measured). 
- **Voiced gate**: the original is only mixed while the speaker is making sound (above `gate_db`, held
  `gate_hold_ms` across short gaps), so room noise is never added. Silent audio is not even queued, so a
  silent speaker keeps the track idle.
- **Ducking**: the original sits at `base_gain_db` (default -10 dB). While translation audio is playing it
  moves to `duck_gain_db` (-20 dB) over `attack_ms` (120 ms) and comes back over `release_ms` (500 ms),
  with linear ramps inside each block (no steps).
- **Limiter**: above 80% of full scale the sum is compressed smoothly (tanh knee), never hard-clipped.
- **Routing**: a listener only ever gets the other participant's voice (`feed_original` maps speaker to
  the other lane), so nobody hears their own delayed voice. The original travels inside the listener's own
  translation track; nobody is subscribed to anyone's microphone, so private listening and the
  server-side isolation of [audio_routing](../runtime/src/runtime/audio_routing.py) keep working. Phones
  work too.
- **Isolation from STT/VAD/translation**: the copy is taken after the lane's frame is resampled and before
  it is pushed; the lane's frame is the same object either way (a test checks it). `feed_original` never
  raises. No change when dubbing is off: the router takes the old direct path and the input tap does
  nothing.
- **Resources**: one small mixer per listener, created when their first translation frame arrives and
  released on unpublish, on `set_dubbing`, and when the call ends (`ListenerAudioRouter.close`). Buffers are
  bounded (original: delay + 2 s, translation: 60 s) and an interruption drops the translation not yet played.

## Measured

All free: synthetic and recorded test audio, LiveKit tones, no providers.

| What | Result | How |
|---|---|---|
| Added latency, mixer idle (translation alone) | about 0 to +2 ms (median, 2 runs of 20 rounds, 1 of 16) | `scripts/dubbing_probe.py`, through LiveKit, rotating with the direct path |
| Added latency, original already playing | +9 to +11 ms (median, same runs) | same; with 20 ms blocks it was +28 ms, hence 10 ms blocks |
| Mixing cost | 17 microseconds per 10 ms block, 0.17% of a core | `scripts/dubbing_lab.py`, synchronous |
| Microphone tap | 6 microseconds per 10 ms frame on the lane's audio task | resample 16 to 24 kHz and queue |
| Timer wake-ups while a listener's mix is active | about 100 per second; on this machine a bare 100 Hz loop costs about 3% of a core | `asyncio.sleep` loop; machine-dependent |
| Memory | about one translation sentence (200 kB) plus the original's delay line (a few kB) | peak in the lab |
| Clipping | 0 samples in every scenario | lab |

Added latency is one sentence's first audio; STT, translation and TTS are not touched. Absolute LiveKit
latencies moved between runs (116 to 209 ms for the direct path); only the within-run differences are
meaningful.

## Offline experiments (`scripts/dubbing_lab.py`; proxies, not listening results)

Recorded benchmark utterances on a virtual clock, scenario "continuous" (the speaker keeps talking while the
previous sentence is translated; the case where the two overlap):

| Profile | presence | translation-to-original ratio median / p10 | step per 10 ms |
|---|---|---|---|
| subtle (delay 0, base -16, duck -26) | 45% | 33.9 / 20.2 dB | 1.15 dB |
| balanced (150, -10, -20) | 66% | 27.8 / 10.0 dB | 1.42 dB |
| present (300, -6, -18) | 73% | 24.9 / 7.2 dB | 1.87 dB |

"presence" is the share of the speaker's voiced time still audible in the mix (above -40 dBFS).

What the sweeps say (balanced, continuous speech):

- **Duck depth is the main lever**: -16 dB gives 74% presence but the translation is only 6 dB above the
  original in the worst 10% of 100 ms windows; -24 dB gives 14 dB and 57% presence. This trade-off is for
  listeners to decide.
- **Attack** does not change clarity (the ratio is the same from 20 to 250 ms) but changes smoothness:
  20 ms steps by 6.6 dB, 150 ms by 1.2 dB. Hence 120 ms.
- **Release** below 300 ms makes the original jump back (2.9 dB steps); 500 ms or more is smooth.
- **Delay** barely changes the ratio or presence in continuous speech (0 to 1200 ms). It changes when the
  speaker's voice appears relative to the translation, which only people can judge.
- **No overlap in a single sentence**: the translation starts about 1.26 s after the speaker stops, so
  for a lone sentence the original has already ended. The effect is felt when the speaker keeps talking, and
  the original's end then fades out rather than cutting (gate release 150 ms).

Rendered WAVs (`--wav-dir`) are the material for a listening study; they are not committed.

## Limitations and unresolved

1. **No listener has heard it.** Intelligibility, naturalness and trust are untested. The numbers above
   are proxies. The default profile is a starting point, not a result.
2. **Echo risk is untested end to end.** Without headphones, the speaker's quiet voice plays from the
   listener's speakers into their microphone and then into their STT lane. Browser echo cancellation should
   remove it; a phone handset may not. The mixer is only a copy and cannot cause it on the speaker side, but
   it can on the listener's. Needs a speaker-phone test with real audio and, if it occurs, gating the
   listener's lane while their mix is loud.
3. **The original is only live, never replayed under the translation.** For a lone sentence it has ended
   before the translation begins (see above). A "voice-over" mode that holds the original and plays it under
   the translation would need to pair utterances with translations; not built.
4. **Continuous 10 ms timer while a mix is active**, with a measured cost that depends on the machine. At
   scale this should be measured on the real deployment; coarser blocks halve it for +17 ms during overlap.
5. **Tested with tones through LiveKit**, not with a browser client or a phone, and not with a real call
   (that needs ElevenLabs credits).
6. **No runtime control path** (per-call toggle from the client) beyond `set_dubbing`; only the session
   setting exists.
7. Video: the original is about 310 ms behind a speaker's image.

## Ready for live user testing?

For an internal listening study with headphones: yes, off by default and safe when off. Before exposing it
to customers: the echo test on speakers and phones, a longer soak for the timer cost, and the listening
study itself.

## Listening demos and evaluation plan

Rendered comparisons of AI-only, the live prototype and two exploratory replay variants, their
measured properties, a telephone-codec check and the listening-study protocol are in
`docs/dubbing-listening.md` (audio in `runtime/bench/dubbing/listening/`).
