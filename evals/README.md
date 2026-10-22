# evals.voice-labs

Dataset, metrics and benchmark runner for Voice Labs speech translation (RM-012).
Everything committed is text: the manifest, code and tests. **No audio and no recordings
are stored in git.**

| Piece | Where |
|---|---|
| Dataset manifest v1 (140 items) | `datasets/v1/manifest.jsonl` |
| Manifest schema, loading, validation | `src/evals/dataset.py` |
| Audio generator (macOS `say`) | `src/evals/generate_audio.py` |
| Metrics: WER, code-switch WER, CER, chrF, terminology recall, p50/p95 | `src/evals/metrics.py` |
| Runner: `stt`, `translate`, `cascade` | `src/evals/run.py` |
| Live latency report (p50/p95 per engine and mode, NS-01) | `src/evals/live_latency.py` |

Run everything from this directory (`evals/`). The commands below use the workspace
Python (`~/.venvs/voice-labs/bin/python`, which has `httpx`); the tests and the generator
need only the standard library plus `pytest`, the runner needs `httpx`.

```bash
cd evals
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider   # no network, no cost
```

## Dataset v1

One JSON object per line. Categories and counts:

| Category | Items | What it covers |
|---|---:|---|
| `en-ko`, `ko-en`, `ru-en`, `en-ru` | 20 each | clinic, immigration, booking, customer-support sentences |
| `mixed` | 20 | Russian with English words (`Мне нужно book an appointment на завтра`), 10 to English, 10 to Russian |
| `names-numbers` | 10 | dates, times, amounts, names, addresses |
| `domain-terms` | 10 | visas, residence permits, appointments, clinic and booking vocabulary |
| `noisy-phone` | 10 | derived from other items: white noise at 5/10/15 dB SNR and/or 8 kHz telephone band |
| `interruptions` | 10 | two utterances in one clip: a request, then a correction that cuts in |

Fields: `id`, `category`, `source_language` (`en`/`ko`/`ru`/`ru+en`), `target_language`,
`text` (exactly what is spoken), `reference_translation`, `key_terms`, `tts_voice`; plus
`derived_from` and `degradation` (noisy-phone) or `followup` (interruptions).

Conventions: spoken `text` contains no digits (numbers are written as spoken words, so WER
is not a number-formatting test); `key_terms` may mix source- and target-language terms,
each is scored only on the side whose text contains it; `a|b` lists acceptable
alternatives (`appointment|reservation`). A Hangul neighbour does not break a term match
(`비자` matches `비자를`). References were written and reviewed by hand, not by a model;
have a native speaker review them before using the numbers for decisions.

## Generate audio (local, free)

```bash
python -m src.evals.generate_audio                      # all 140 -> ~/.cache/voice-labs-evals/v1
python -m src.evals.generate_audio --out /path --categories mixed,en-ko --limit 5
python -m src.evals.generate_audio --voice ko=Yuna --force
```

16 kHz mono WAV per item (`<id>.wav`) plus `audio_manifest.jsonl` (duration, voices,
degradation). Default voices: `Samantha` (en), `Milena` (ru), `Yuna` (ko). Code-switched
text is split by script and the segments are concatenated; interruption items are the first
utterance, a pause, then the interrupting utterance. These are **synthetic voices**: good
for regression and relative comparisons, not a measure of real-speaker accuracy. The audio
folder is outside the repository (and `evals/.gitignore` also excludes `audio/`,
`results/`, `*.wav`). Full set is about 10 minutes and renders in under two minutes.

## Run the benchmark (billable)

The runner calls Deepgram and OpenAI and refuses to start without `--confirm-billable`.
Load keys in your shell first (never commit them):

```bash
set -a; source ../../stack.voice-labs/.env; set +a      # DEEPGRAM_API_KEY, OPENAI_API_KEY
python -m src.evals.run cascade --confirm-billable --limit 3          # 3 per category
python -m src.evals.run stt --confirm-billable --categories mixed --stt-language multi
python -m src.evals.run translate --confirm-billable --models gpt-4.1-mini
```

| Mode | Does | Provider |
|---|---|---|
| `stt` | transcribe audio; WER, CER (Korean), code-switch WER by script, terminology recall | Deepgram prerecorded `nova-3` |
| `translate` | translate the gold text with the strict interpreter prompt; chrF, term recall, TTFT/total | OpenAI chat (`gpt-4.1-nano`, `gpt-4.1-mini`; `--models` to change) |
| `cascade` | STT then translation of the transcript; per-stage latency and end-to-end | both |

Options: `--limit N` (per category), `--categories a,b`, `--stt-language fixed|multi`
(`fixed` uses the spoken language and `multi` for code-switched items; Korean always uses
`ko`), `--stt-model`, `--audio-dir`, `--out`. Output goes to
`~/.cache/voice-labs-evals/results/<run-id>/`: `results.jsonl` (one record per item and
model, including transcripts, translations and all scores) and `summary.md` (per-category
table plus run metadata: dataset version and hash, models, date, machine).

Cost note: the full cascade over all 140 items is about 10 audio minutes of Deepgram
(a few cents) plus 280 short OpenAI calls with the two small models (cents). A
`--limit 3` run is a fraction of a cent. The runner prints the request plan before it
starts. Latency is the wall time of one request from this machine (sequential), so it
includes network; it is not a streaming-STT or TTS number.

## Live latency report (billable, needs running servers)

`live_latency.py` plays dataset audio into live interpreter calls (Platform + Runtime +
LiveKit running) and reports p50/p95 time to first translated audio per engine x latency
mode, both from the Runtime's `utterance_timeline` events and as heard by the simulated
listener, plus missing translations. NS-02/NS-03 columns are placeholders.

```bash
python -m src.evals.live_latency --dry-run                       # plan and cost, no network
python -m src.evals.live_latency --platform http://localhost:8300 --confirm-billable \
    --engines realtime,cascaded --modes instant,balanced,accurate --pairs ko-en \
    --utterances 30 --out report.md [--record-dir DIR] [--two-person]
```

Needs `livekit` and `httpx` (the workspace Python). The unit tests cover the plan, speech
and onset detection, percentiles and the report with no network. See the module docstring
and `stack.voice-labs/docs/evaluation.md`.

## Add real, consented recordings later

Keep the manifest fields as they are. Put `<id>.wav` (16 kHz mono) recordings for the same
ids into the audio folder (or `--audio-dir`) and run the same commands; the runner only
needs the file. Add new items as a new dataset version (`datasets/v2/manifest.jsonl`) rather
than editing v1, so results stay comparable. Only add recordings with the speaker's
consent, and keep the audio itself out of git.

## Metrics

- **WER** word error rate (casefolded, punctuation-insensitive, NFC, `ё`=`е`).
- **Code-switch WER**: the same alignment, with each error charged to the script of the word
  that failed (Latin / Cyrillic / Hangul). Exposes cases such as `book` becoming `бук`.
- **CER** character error rate over letters and digits; used for Korean sources.
- **chrF** character n-gram F-score (n = 1..6, beta = 2, 0-100, whitespace ignored,
  case-sensitive); `word_order=2` gives chrF++. It is a similarity measure, not a verdict:
  a correct paraphrase scores below 100.
- **Terminology recall**: share of `key_terms` found in the output, among terms present in
  the reference.
- **Latency**: p50 / p95 (linear interpolation) of seconds.
