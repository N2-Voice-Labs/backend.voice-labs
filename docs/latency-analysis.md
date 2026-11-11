# Latency analysis (offline, from the recorded benchmark runs)

Source data: `runtime/bench/` (two-person ko/en call, ElevenLabs STT + Google Cloud Translation + ElevenLabs TTS,
the same 37 recordings in Balanced and Instant, plus 1,000 direct Google requests and four small fault runs).
Reproduce with `cd runtime && python scripts/benchmark_analysis.py`. No provider is called.
Nothing here changes a production setting; every proposal below needs its own measurement first.

Sample sizes are small (37 sentences per profile, 5 per fault run). P99 and rare-event rates are indications, not results.

## 1. Where the ~1.28 s goes (unsplit sentences, last spoken sound to first translated audio)

Correction to the first version of this document: Pipecat's `stt_ttfb` is measured from the VAD's estimate of the last spoken
sound (VAD stop minus `stop_secs`), so it already contains the 200 ms VAD silence. The first version added the two and
therefore reported ~187 ms as unexplained; the real unexplained part is about **370 to 390 ms**.

| step (medians, from the last spoken sound) | Balanced | Instant |
|---|---|---|
| STT final transcript (includes the 200 ms VAD silence) | 459 ms | 470 ms |
| not covered by any mark, after the final transcript | 387 ms | 373 ms |
| = before the translation starts | **846 ms** | **851 ms** |
| translation (first chunk) | 262 ms | 269 ms |
| TTS first audio | 183 ms | 181 ms |
| = end to end | 1,289 ms | 1,283 ms |

- **What the code says about the turn pause.** Pipecat's `SpeechTimeoutUserTurnStopStrategy` ends the turn when both are done: the pause
  timer, started at the VAD stop (so it ends 200 ms + pause after the last sound), and the first final transcript. Balanced: pause
  ends at 500 ms, STT median 459 ms, so the pause is the later one for almost every sentence, by about 40 ms. Instant: 400 ms,
  never the later one. The recorded data cannot show a 40 ms difference (spread ~40 ms, n = 29 and 17), so this is the code's
  prediction, not a measurement. The first version of this document wrongly said the pause was completely hidden by the STT.
- The steps are sequential: Google returns a whole sentence and TTS needs the text.
- Translation as seen by the pipeline (262 ms) is ~95 ms above the direct Google call (P50 168 ms).

## 2. Premature splits

- Every Balanced split is also an Instant split (5 of 13), and the longest silence *inside* the recorded sentence predicts them:

| | split | not split |
|---|---|---|
| Balanced (pause 0.3 s) | inner silence 0.25 to 0.38 s | at most 0.24 s |
| Instant (pause 0.2 s) | 0.08 to 0.38 s (11 of 13 at 0.18 s or more) | at most 0.20 s |

  A silence of length g splits the sentence when g + VAD start (0.2 s) is longer than VAD stop (0.2 s) + turn pause, that is, when g
  is about the turn pause. Instant's lower pause cuts at ordinary breath/comma pauses.
- Harm, judged from the stored STT text and translation:
  - Balanced: 5 splits, **0 harmful**. All were `저기요.`, `혹시 …은…` style clause breaks; both halves translated correctly.
  - Instant: 13 splits, **6 harmful**. In 5 of them the STT invented `Mm-hmm`/`음.` or dropped words at the cut (`translate- Mm-hmm | ... into Korean`
    lost "this document"; `By the way, can- | Mm-hmm ... you explain` was translated as two broken pieces). In 1 a one-word fragment (`그리고-`)
    was spoken alone. The other 7 are benign.
- The two Instant splits with short gaps (u0002 at 0.12 s, u0007 at 0.08 s) are not explained by the gap; both are the STT adding a filler.
- **Raising the Balanced pause from 300 to 400 ms is not free.** The turn ends at 200 ms + pause after the last sound, or when the
  final transcript arrives, whichever is later (median 459 ms). At 0.3 s that is 500 ms; at 0.4 s it is 600 ms, so every
  sentence waits about **+100 ms**, not zero. (The first version of this document called it free; that rested on the double-counting
  error described in section 1.) What it would buy, replaying the recorded silences with the rule "split when the silence is longer
  than about the pause": the 5 Balanced splits at 0.25 to 0.38 s drop to about 2 (u0028 at 0.38 s, u0000 at 0.35 s; u0014 at 0.34 s is on the
  edge), and to 0 at 0.5 s (+200 ms). Those 5 splits were all benign, so on this corpus the trade is ~100 ms of latency for no quality
  gain. Not recommended; production keeps 0.3 s. The threshold fit comes from 37 synthetic sentences; real speakers pause differently.

## 3. Google deadline (1,000 direct requests, real endpoint)

Google P50 168 ms, P99 461 ms, max 1,351 ms. Gemini whole-sentence time: median 946 ms (713 to 1,373 ms, **8 samples**, from the
fault runs).

| deadline | false timeouts per 1,000 | 95% interval | cost of one false timeout | mean cost per sentence | wait when Google hangs |
|---|---|---|---|---|---|
| 500 ms | 6 | 0.28 to 1.30% | ~3.8 s summed | 23 ms | 1.45 s |
| 750 ms | 1 | 0.02 to 0.56% | ~3.5 s summed | 3.5 ms | 1.70 s |
| 1,000 ms | 1 | 0.02 to 0.56% | ~3.7 s summed | 3.7 ms | 1.95 s |
| 2,000 ms (current) | 0 | 0 to 0.38% | n/a | 0 ms | 2.95 s |

- "Cost of one" is summed over the sentence that timed out and the next 4 that go to Gemini during the 30 s cooldown (each about
  0.8 s slower than Google), not a single stall.
- Trade: a 750 ms deadline saves about 1.25 s for each hung Google request compared with 2,000 ms and adds about 3.5 ms per sentence
  on average from false timeouts. It is cheap only if Google really is as steady as in this run.
- Limits: one 5-minute burst from one machine; no outage, time-of-day or regional variation; a genuine hang is not in the data; Gemini has 8
  samples; the cooldown length (30 s) and 4 sentences per cooldown are assumptions; translation differences between engines are ignored.
  The fault runs (750 ms deadline with a 1,000 ms delay) behaved as the model predicts: 2 fallbacks, no frozen call.
- Production keeps 2 s until a longer or repeated run backs a lower value.

## 4. Instrumentation added (no behaviour change)

New marks on the utterance timeline, all on the Runtime's monotonic clock, in the order they happen:

| mark | taken when | step it closes |
|---|---|---|
| `speech_end` | estimated: `vad_stop` minus the VAD's `stop_secs` | |
| `vad_stop` | the VAD frame reaches the translator | `speech_end` to `vad_stop` is the fixed 200 ms |
| `final_transcript` | first finalized transcript of the turn reaches the translator | STT (network: ElevenLabs) |
| `turn_end` | the turn-stop strategy ended the turn | wait for the pause timer, overlapping the STT wait |
| `translation_start` (existing) | translation request about to be sent | in-process hop |
| `translation_first_chunk` (existing) | translated text received | network: Google |
| `tts_request`, `first_audio` (existing) | text handed to TTS, first voiced audio | network: ElevenLabs |

- The event also carries `t0_monotonic_s` and the Runtime logs one line per timeline (`utterance_timeline … t0_monotonic_s=… marks_ms=…`).
  Linux's `CLOCK_MONOTONIC` is shared by all processes of a host, so a client on the same host can line its own `time.monotonic()` stamps up
  with them. `translation_benchmark.py` now records `mono_start`, `mono_end_speech` and `mono_first_audio` and, with `--runtime-log`,
  prints a **fine timeline** table (uplink+detect, vad_silence, stt, pause_wait, hop, translation, tts, downlink). Its first live run is section 8.
- Clock audit: every duration the Runtime computes uses `time.monotonic()` (the tracker's default clock) or the event loop clock. A test
  (`test_no_latency_code_reads_the_wall_clock`) fails if `time.time()` or `datetime.now()` appears in any Runtime module except the
  event/health modules, which stamp records for people. Pipecat itself stamps its VAD frames and computes `stt_ttfb` with the wall clock;
  the new marks deliberately ignore the frame stamps and take the time when the frame reaches the translator.
- Sequential versus overlapping: VAD silence, STT, turn end, translation, TTS and the output are sequential. The pause timer overlaps the
  STT wait. Network-dependent: uplink, STT, translation, TTS, downlink. In-process: VAD analysis, turn end, the hop to the translator.

## 5. Translating from partial transcripts: feasibility (analysis only, nothing built; measured result in section 8)

- **What exists:** ElevenLabs' realtime STT sends partial transcripts while the person speaks; the pipeline turns them into
  `InterimTranscriptionFrame`s that today only mark `first_transcript`. They carry no stability flag and no word timings in this path,
  and the partial text was not stored in the recorded runs, so how often the last partial equals the final is **unmeasured**.
- **What not to do:** translate partials during speech. Korean puts the verb last and English-to-Korean reorders the sentence, so a
  translation of half a sentence is wrong or meaningless and changes with every word. Only the partial at the end of speech is a candidate.
- **Candidate design (speculate at the VAD stop):** at `vad_stop` (200 ms after the last sound, ~260 ms before the final transcript) take the latest
  partial and start the translation. When the final transcript arrives, compare the two after normalising case, punctuation and spaces. Equal:
  keep the speculative result. Different: discard it and translate the final. Savings: the Google call overlaps the ~260 ms STT finalization and
  the remaining pause (up to 262 ms on a hit).
- **No duplicate translation:** one translation job per turn, keyed by the normalised source text. A speculative result is held in the job and
  released only if the key of the final text matches; a miss cancels the in-flight request and starts the real one. The conversation
  history, the transcript messages and the `translation_failures` counters are written only for the released result.
- **No premature speech:** nothing is handed to TTS before `turn_end` and a matching final transcript. If the speaker resumes (a split is
  prevented by the same pause rule as today), the speculative job is cancelled. This keeps the speaking rule unchanged: only a confirmed turn is voiced.
- **ElevenLabs cost:** unchanged, because TTS runs only on the released text; a missed speculation costs one Google request, not TTS characters.
  Google Translation costs about $20 per million characters, so one extra request of ~44 characters is about $0.0009 per sentence
  (roughly $0.9 per 1,000 sentences). Limit: one speculative request per turn. Prefetching TTS on a hit would save a further ~180 ms but spends characters on misses and
  is not proposed.
- **Risks to accuracy:** low if the release rule is strict (identical normalised text), because then the output is the text the
  non-speculative path would translate. A looser rule (similar enough) would speak a translation of text the STT later corrected. Glossary
  sentences go to the LLM and the speculative path must keep that route.
- **Preconditions before building:** (1) log partial and final text of a funded run to measure the hit rate; if it is under ~60% the gain is
  negligible; (2) the fine timeline, to confirm the Google call really overlaps something that would otherwise be idle.

## 6. Expected end-to-end savings (medians, unverified estimates)

| option | expected saving | complexity | risk to accuracy | note |
|---|---|---|---|---|
| Eliminate avoidable pipeline overhead (the ~380 ms after the final transcript, the ~95 ms above the raw Google call) | 100 to 250 ms; depends on how much of the 380 ms is uplink that cannot be removed | low once measured: marks are added; fixes are local | none | unknown split until the next funded run |
| Reduce STT finalization (lower VAD `stop_secs`, a shorter pause, earlier commit) | 40 to 100 ms | low (settings) | medium: every 100 ms less pause moves the split threshold by about 100 ms; STT may invent fillers on short fragments (seen in Instant) | the Balanced pause alone binds by ~40 ms; changing only it recovers ~40 ms |
| Translate from the last partial at the VAD stop | up to ~260 ms on a hit (hit rate unknown) | high: speculative job, cancel and release logic, tests | low with a strict match, medium if loosened | needs the hit rate first |

## 7. Recommendation, in order

1. **Run the fine timeline once (about 450 ElevenLabs characters for ~10 sentences, or the full set after the quota resets).** It is the only
   way to split the ~380 ms; cost is tiny, risk none. Then remove whatever in-process time it shows.
2. **Measure the partial-versus-final agreement in the same run** (log both texts). If the hit rate is high, build the speculative translation
   behind a flag (design in section 5). It is the only option whose saving can exceed 200 ms, and it keeps the accuracy rule unchanged.
3. **Leave the STT/pause settings alone.** The 40 ms available from the Balanced pause is not worth the shifted split threshold; the 400 ms pause
   costs +100 ms for a quality gain that this corpus does not show.

Production defaults are unchanged: Balanced, Google Translation, 2 s deadline, 0.3 s pause.

## 8. Live fine-timeline run (10 sentences, 343 ElevenLabs characters)

`bench/fine_timeline/` (raw data), Balanced, Google Translation, 5 Korean and 5 English sentences from 15 to 68 translated characters. Exploratory:
n = 10 (9 in the table; u0018 had two STT segments in one turn, which makes the first-final mark meaningless for it).
Quota afterwards: 9,703 of 10,000 used, 297 left.

**Clocks.** The benchmark client and the Runtime both use `time.monotonic()`, which is `clock_gettime(CLOCK_MONOTONIC)` on this host
(Python 3.12.3 in both virtualenvs). It is one system-wide clock for all processes: a child process read 1.7 ms before the parent on the
same call (the cost of spawning it), and `clock_gettime` agreed with `time.monotonic()` to 3 microseconds. Valid only on one host; the Runtime
refuses nothing if it is not, so a cross-host run would give nonsense offsets.

**How `speech_end` is derived.** `speech_end = vad_stop - stop_secs`, where `vad_stop` is the moment the VAD's stop frame reaches the translator
(taken on the monotonic clock; the frame's own stamp is wall-clock and ignored). Its error is not calibrated, because there is no ground truth inside
the Runtime: (a) Silero decides in 32 ms windows, so the decision lags the real silence by up to ~32 ms; (b) Silero counts a frame as silence when its
confidence falls below 0.7 or the volume below the minimum, which can happen before the last audible sound has ended (the estimate is then early);
(c) queueing between the VAD decision and the translator adds delay to `vad_stop` (the estimate is then late). Rough bound: tens of
milliseconds, not hundreds. It is also not a pure uplink measurement: the step `uplink+detect` below mixes real transport with these effects.
Its spread across sentences is small (136 to 202 ms, standard deviation 19 ms), which points at a constant, not at jitter.

**Per-stage breakdown** (median [min to max] ms, 9 sentences; the medians add up to 1,261 ms against a measured 1,264 ms, and each sentence's
steps add up to its own end-to-end time exactly):

| step | ms | what it is |
|---|---|---|
| uplink + detect: last sound at the client to `speech_end` at the Runtime | 161 [136-202] | **transport / buffering** (client send queue, LiveKit, Runtime input buffer) plus VAD effects above |
| VAD silence | 200 [fixed] | **required waiting**: `stop_secs` |
| STT: VAD stop to first final transcript | 283 [219-312] | **provider processing and network** (ElevenLabs) |
| pause wait: final transcript to turn end | 19 [-3 to 82] | **required waiting**, almost fully hidden behind the STT (as predicted: ~20 to 40 ms) |
| hop: turn end to translation request | 0 [0-4] | **in-process overhead: none** |
| translation: request to first chunk | 257 [240-306] | **provider processing and network** (Google), ~85 ms above the direct call |
| TTS: text to first voiced audio | 192 [178-230] | **provider processing and network** (ElevenLabs) |
| downlink: first audio at the Runtime to the client hearing it | 149 [119-199] | **transport / buffering** (output pacing, LiveKit, client playout) |
| **end to end** | **1,264** | |

The earlier "unexplained ~380 ms" is: uplink+detect 161 + downlink 149 + pause wait 19 + hop 0 = 329 ms of transport, buffering and waiting, plus the
~50 ms by which this sample's STT (283 + 200 = 483 ms from the last sound) is slower than the earlier 37-sentence run (459 ms). **Avoidable in-process
pipeline overhead is essentially zero**: the hop is under 1 ms and the pause costs ~20 ms.

**Partial versus final transcript** (11 STT segments in 10 sentences; the partial is the latest one before the VAD stop):

| outcome | segments |
|---|---|
| no partial yet | 1 |
| exact match with the final | **0** |
| equal after case, punctuation and spacing normalisation | **0** |
| partial is only the beginning of the final (last words missing) | 8 |
| other (different word, cut mid-word) | 2 |

- The latest partial is already old at the VAD stop (median 541 ms, up to 992 ms) and misses the end of the sentence: `... 18,6` for `18,677 won`,
  `... look at the contract.` for `... contract before I sign it.`, a Korean partial cut inside a word. The final arrives 283 ms (219 to 312) later and carries the
  missing words.
- **Match rate 0 of 11** (exact and normalised). With 11 samples the true rate could still be up to ~25%, but every observed miss lost real content (digits, a clause,
  a verb), so a translation started from the partial would have been wrong, not just differently punctuated.
- Realistic saving of speculative translation from the partial at the VAD stop with this STT: **about zero**. A looser rule that accepts prefixes would speak
  incomplete translations. The idea is not worth building with this STT.

**Idle connection to Google (free test, `scripts/idle_connection_probe.py`).** The Runtime's Google client is httpx's default, whose keep-alive expires after 5 s idle.
Called in isolation with 6 to 7 s between sentences, the Runtime translator took 444 and 464 ms (median) against 232 and 238 ms when the client keeps connections 60 s,
and 243 ms with 2 s gaps. In the live call, however, the translation step was 232 to 306 ms with 7 to 10 s between requests, so the call did not show the cold-connection cost.
Why is unexplained (the isolated test and the call use the same class and the default client). The ~85 ms gap between the live step (257 ms) and the hot direct call (~170 ms)
remains unexplained as well. Treat the keep-alive cost as a plausible risk, not a measured loss in the call.

## 9. Highest-value optimization given the measured timeline

Ranking what the 1,264 ms is made of, and what can be changed:

| block | ms | changeable? |
|---|---|---|
| required waiting and provider time: VAD 200, STT 283, translation 257, TTS 192, pause 19 | 951 | only by trading quality (VAD, pause) or switching providers |
| transport and buffering: uplink+detect 161, downlink 149 | 310 | yes, in part: sizes of the input/output buffers, region of Runtime vs LiveKit, jitter buffers |
| in-process overhead | ~0 | nothing to gain |

**Single highest-value optimization: the audio transport and buffering path (310 ms, 25% of the total).** It is the largest block that is neither a provider's
time nor a quality-vs-speed setting, and the only one with no accuracy cost. The first step is to split it, because 310 ms is far above a typical network round
trip: add one mark when the last voiced input frame arrives at the Runtime and one when the first voiced output frame is written to the LiveKit track, then look at
the output pacing and chunk sizes, the LiveKit region and the Runtime's location, and the client's jitter buffer. Realistic saving: 60 to 150 ms. Caveats: the benchmark
client adds its own buffering (`capture_frame`, 10 ms framing, the energy test), so part of the 310 ms is a property of the test, and a browser client will differ.
Speculative translation (0 of 11 matches), a shorter pause or a different turn rule are all worse choices on this evidence.

## 10. Audio transport and buffering

Everything marked **measured** comes from free probes (`runtime/scripts/transport_probe.py`, `idle_connection_probe.py`, raw data in `runtime/bench/transport*/`). **Hypothesis** means not yet confirmed. The new marks (`audio_in_last_voiced`, `tts_first_frame`, `livekit_submit`, `livekit_submitted`) are only recorded with `RUNTIME_TIMELINE_DIAG=1` and have not yet been run on a live call.

### Sources of transport latency

| Source | Result | Status |
|---|---|---|
| LiveKit SDK one-way floor (publisher `capture_frame` to subscriber first loud frame, no speech providers) | about 100-110 ms | measured |
| Network round trip to LiveKit Cloud | about 52 ms (about 26 ms one way) | measured |
| Rest of each 150-160 ms leg (the call's uplink 161, downlink 149) | LiveKit media stack, about 45-55 ms of Runtime, VAD and client effects | hypothesis |
| Paced writes (sleep one chunk between frames) | +30-60 ms; the current burst write is right | measured |
| Chunk size, queue size, DTX | within noise | measured |
| 8 s idle before the first frame | +150-180 ms in one direction only; not seen in the call | measured, unexplained |

The 310 ms of uplink+downlink is therefore not avoidable overhead: about 200-220 ms is the transport floor.

### LiveKit audio pipeline (code inspection)

`rtc.AudioSource(24000 Hz, 1 channel, queue_size_ms=1000)`, 40 ms chunks, per-listener tracks (`ListenerAudioRouter`). Pacing comes from `capture_frame` back-pressure; there are no sleeps before the first chunk, so no artificial output delay.

### Regions

The LiveKit project is restricted to Asia/Middle East regions (nearest oosaka1a 872 km, otokyo1a 1202 km). The Runtime ran locally (timezone Asia/Seoul). TCP connect about 52 ms, TLS about 108 ms. Cross-region cost cannot be judged for a deployed Runtime: it does not exist yet. Deploy it in the nearest of those regions.

### Google Translation connection lifecycle

One persistent `httpx.AsyncClient` per translator, with default limits (keep-alive expiry 5 s). `warm_up()` runs once at call setup. Between sentences (about 6.5 s apart) the connection has expired, so each sentence pays a new TCP+TLS handshake: measured about 250 ms cold against 122-127 ms warm, which matches the live 257 ms translation step.

Long-idle test (keep-alive expiry 120 s, gaps 30/55/90 s): no errors in 12 requests. Latencies were 259, 164, 147, 484, 261 (30 s), 192, 370, 749, 191 (55 s), 214, 213, 239 (90 s). The server may close idle connections silently, and httpx then reconnects, which would explain latencies above the 125 ms warm figure; these samples are too few and too noisy to say that 60 s works, and I did not trace connection opens in that test. **A 60 s keep-alive is safe in the sense of no errors, but its benefit beyond about 30 s is unproven.** Production settings are unchanged.

### Feasible optimizations (estimates, none applied)

1. Keep the Google connection warm: raise keepalive_expiry to about 30 s, or send a cheap periodic request. Expected saving 85-125 ms per sentence (measured cold/warm gap; live confirmation pending).
2. Deploy the Runtime in a LiveKit-adjacent Asian region: the effect depends on where the Runtime goes; hypothesis.
3. Transport floor: no feasible change; paced writes and smaller chunks do not help.

### 10a. Keep-alive comparison (measured, `runtime/scripts/keepalive_comparison.py`)

The Runtime's own `GoogleTranslateTranslator`, identical request, seeded 6-8 s gap before each one, 10 alternating blocks of 10 per configuration (n=100 each), no ElevenLabs credits:

| keepalive_expiry | P50 | P95 | P99 | new TCP+TLS | errors |
|---|---|---|---|---|---|
| 5 s (httpx default) | 455 ms | 727 | 878 | 100/100 | 0 |
| 30 s | 223 ms | 385 | 1295 | 0/100 | 0 |

The reuse hypothesis is confirmed: with 5 s every request opened a connection; with 30 s none did. P50 saving 232 ms (both halves of the run agree: 438 to 212 ms, 465 to 224 ms). The 30 s P99 comes from two requests of about 1.3 s on a reused connection (no error, under the 2 s deadline); with n=100, P99 is the second-worst sample, so it is not a stable estimate. The absolute numbers are larger than the earlier 250 ms cold figure (network conditions differ); the live translation step measured 257 ms. Implemented as `GOOGLE_TRANSLATE_KEEPALIVE_S` (default 30; 5 restores the old behaviour). No periodic requests.

### 10b. Larger keep-alive comparison (measured, n=500 per configuration)

Same method as 10a, 4 parallel lanes of alternating blocks of 25, seeded 6-8 s gaps, the translator's 10 s test timeout (not the production 2 s). Raw rows, with connection age, idle time and phase times, are in `runtime/bench/keepalive/keepalive_500.json`.

| keepalive_expiry | P50 | P90 | P95 | P99 | max | new TCP+TLS | errors |
|---|---|---|---|---|---|---|---|
| 5 s | 463 | 649 | 741 | 1402 | 2397 | 500/500 | 0 |
| 30 s | 205 | 303 | 363 | 454 | 653 | 0/500 | 0 |

- P50 saving 258 ms; the tail improves too (P99 1402 to 454 ms).
- The two ~1.3 s requests on reused connections in 10a did not reproduce: none of 500 reused-connection requests exceeded 653 ms. They were an isolated spike, most likely on Google's side.
- Connection age does not predict latency (correlation with latency 0.03; age 0-30 s P50 216, 30-60 s 202, 60-120 s 204, 120-400 s 204 ms; connections lived up to 176 s). Idle time (6-8 s) also does not (0.05). A reused request spends about 3 ms before sending and about 200 ms waiting for Google.
- The slow 5 s requests are slow in connecting (`to_send_ms` up to 1.9 s for TCP+TLS) or in the first response (up to 1.5 s). One 5 s request took 2.4 s, longer than the production 2 s deadline: with the old behaviour such a handshake spike would have sent that sentence to the Gemini backup.

### 10c. One translator per call (corrected)

`build_translator` runs once per pipeline (`translation_engine.py`, `pipeline.py`), so a call has one translator and one HTTP client shared by all participants; a new call opens a new connection.

Correction to the earlier version of this section: **the `/translations` path never called `warm_up()`** (only the agent-translation path did), so the first sentence of every call paid a cold connection, and no translator client was ever closed. Both are now fixed: `TranslationProcessor` warms the translator when the pipeline starts and closes it in `cleanup()` (the agent-translation processors close it too; closing twice is harmless). Clients passed in from outside are left to their owner. `scripts/connection_leak_check.py` (free, local server) simulates 40 calls: without the close, 11 client connections were still open afterwards (5 after a forced garbage collection); with it, none. A pause longer than the keep-alive between two sentences still costs one reconnect.

### 10d. Concurrent calls (audit and test)

Audit of module-level mutable state in the Runtime: the provider registries (`speech_providers`, `translation_engine`) are filled once at import and never written during a call; per-session state in `api.py` is keyed by session id, removed when the session ends, the finished-session history is capped at 1024, and `max_sessions` caps simultaneous calls. Everything else a call touches (translator and its HTTP client, the Google-to-Gemini fallback cooldown, history, timeline) lives on objects built per call. `test_concurrent_calls_do_not_share_translator_state` runs eight calls at once with interleaved timing, odd-numbered ones with a failing primary: each call speaks only its own translation (a primary outage and its cooldown stay inside that call) and each closes exactly its own translator once. Not tested: real load (many live LiveKit rooms); the per-call cost is one HTTP client and, with the 30 s keep-alive, at most one idle connection per call.

## 11. Two-listener calls: `first_audio` used one playout clock for both listeners (found 2026-11-07, fixed in the working tree)

**Symptom.** In the first real two-phone call, one turn (turn 11, Korean "이게 뭐야?" for the Korean listener) reported
`first_audio = 3053 ms` against `tts_first_frame = 974 ms`; the turn before it (turn 9, an English translation for the other
listener) had `first_audio = tts_first_frame = 1180 ms`.

**Cause (code, not a log).** `TimelineTracker` kept one `_queue` and one `_playhead` for the whole pipeline, and
`first_audio` is `max(arrival, playhead)` on that simulated clock, not a measurement. In a two-person call each
listener has their own TTS, output destination and LiveKit track, so one listener's long translation moved the other listener's
simulated playhead. Rebuilt from the recorded marks, turn 11's `first_audio` fell exactly 2.650 s after turn 9's, the plausible length of
turn 9's English audio (the log with the audio lengths was lost, so this is consistent, not proven). The `livekit_submit < tts_first_frame` order
of the same turn comes from the same lack of ownership: `_head()` was the front of one queue, so a chunk written to one listener's track could mark
the other listener's utterance. Whether that is the whole explanation of 781 < 974 ms is not established.

**Change.** The tracker now has one queue and one playout clock per output destination
(`transport_destination`, the listener's track). TTS frames, `TTSStoppedFrame` and the hand-over to LiveKit are matched by the destination
they carry, and the interpreter registers each utterance with its listener's destination. Single-output engines use the destination `None` and are
unchanged. The diagnostic line now names `speaker`, `listener` and `destination`.

**Tests** (`runtime/tests/test_timeline.py`): a 2.65 s translation for one listener does not move the other's first audio (the old result was
+2.65 s); the same listener still waits for their own earlier audio; one destination's stop frame never ends the other's utterance; a chunk
written to one track never becomes the other's `livekit_submit`; frames carry their destination through `RuntimeEventProcessor`; single-output behavior is
unchanged; the diagnostic line carries the identities.

**Not yet measured.** The corrected number for the call above (about 974 ms server-side) is an estimate from the fixed arithmetic, not a new
measurement; nothing was measured in the browser. For the next real call, `scripts/live_call_stack.sh` keeps every log under
`runtime/bench/live/<timestamp>/` and `summary` prints each turn with speaker, listener and destination; a browser-side
first-audible timestamp for each phone is still to be added, and is what would confirm the corrected number.
