# Latency analysis (offline, from the recorded benchmark runs)

Source data: `runtime/bench/` (two-person ko/en call, ElevenLabs STT + Google Cloud Translation + ElevenLabs TTS,
the same 37 recordings in Balanced and Instant, plus 1,000 direct Google requests and four small fault runs).
Reproduce with `cd runtime && python scripts/benchmark_analysis.py`. No provider is called.
Nothing here changes a production setting; every proposal below needs its own measurement first.

Sample sizes are small (37 sentences per profile, 5 per fault run). P99 and rare-event rates are indications, not results.

## 1. Where the ~1.28 s goes (unsplit sentences, last spoken sound to first translated audio)

| step | Balanced | Instant | waits for the previous step? |
|---|---|---|---|
| VAD stop (fixed `stop_secs`) | 200 ms | 200 ms | yes |
| STT final transcript after the commit | 459 ms | 470 ms | yes |
| not covered by any mark | 187 ms | 173 ms | n/a |
| = before the translation starts | **846 ms** | **851 ms** | |
| translation (first chunk) | 262 ms | 269 ms | yes |
| TTS first audio | 183 ms | 181 ms | yes |
| = end to end, median | 1,289 ms | 1,283 ms | |

- **The turn pause is not on the critical path.** Pipecat's `SpeechTimeoutUserTurnStopStrategy` ends the turn when both the pause
  timer (0.3 s / 0.2 s, started at the VAD stop) and the STT transcript are done. STT needs at least 410 ms (median 460 ms), which
  is longer than either pause, so they run in parallel and the pause costs nothing. That is why Instant is not faster: the data
  shows the same 846 vs 851 ms before the translation.
- Everything else is sequential: nothing starts before the step above it ends, because Google Translation returns a whole
  sentence (no partial output) and TTS needs the text.
- The ~180 ms "not covered" is transport (audio in over LiveKit, audio out to the listener), VAD frame analysis and the hop from the
  STT frame to the translator. It cannot be split further without a new mark.
- Translation as seen by the pipeline (262 ms) is ~95 ms more than the direct Google call (P50 168 ms). Part is the request
  overhead inside the process, part the chunker/context; not measured separately.

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
- **Hypothesis (not tested, so no setting changed):** the turn pause can be raised from 0.3 s up to about 0.4 s with no latency cost,
  because the STT wait (410 ms minimum) already hides it. At 0.4 s the longest observed inner silence (0.38 s) would no longer split.
  Test: replay only the ~10 sentences with an inner silence of 0.18 s or more (about 450 ElevenLabs characters), then the full set.
  Limits: this corpus uses clean synthetic speech; real speakers pause longer and unevenly, and a longer pause also delays the
  sentences that are spoken with a short STT wait.

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

## 4. Largest remaining opportunities

Estimates are for the median sentence and are unverified; each is a hypothesis for the next measured run.

1. **STT final wait, ~460 ms (the largest block).** The commit only happens after the VAD stop, then ElevenLabs needs ~410 to 500 ms.
   Options: commit earlier or on a shorter silence, or translate speculatively from a stable partial transcript and replace it only if the final
   differs. Estimated saving 100 to 250 ms. Risks: a shorter silence raises splits (see 2); speculation can speak text that then changes
   and doubles the spend of translation and TTS characters.
2. **The ~180 ms no mark covers, plus ~95 ms of translation overhead above the raw Google call.** First add marks (frame received, VAD
   stop seen, final text handed to the translator, first audio frame written), then act. Estimated saving 50 to 150 ms. Risk: low for
   marks; unknown for the fix until the split is known (it may be transport time that cannot be reduced).
3. **Fallback path.** One false or real Google timeout puts the next ~30 s on Gemini at ~0.95 s per sentence, 0.8 s slower than
   Google. A shorter cooldown (for example 10 s) or a cheap Google probe would return to Google sooner: saves ~2 s per event but only
   0.1 to 0.6% of sentences, so under 25 ms on average. Risk: flapping between engines if Google is genuinely unstable.

Not an opportunity: the Instant profile. It changes neither the median nor the time before the translation, and it splits more.
