# Costs, speed and voice options

Status: research and measurements, 2026-11-14. Written for the business plan: what a call costs us,
how fast it is, and which provider options exist. Measured numbers come from our own runs (see
[Status](status.md) for the evidence and the credit ledger); market figures are marked as third-party
and must be re-checked before they go into a pricing slide.

## What the MVP does today (measured on real providers)

| Capability | Result | How measured |
| --- | --- | --- |
| AI admissions voice assistant | 10/10 calls answered; median **2.2 s** from the end of the question to the first spoken word of the answer | 10 consecutive calls, 2026-11-13 |
| Answer accuracy (Inha admissions FAQ) | **20/20** factually correct, nothing invented; questions it cannot answer are handed to a human | 20-question set built from Inha's official guidelines |
| Hand-off to a human officer + live translation | officer joined in ~6 s; **0.84 s** median from the end of a sentence to the translated voice | two-browser call, English ↔ Korean, 11 sentences |
| Either person may speak either language | each sentence's language is detected; correct on 6/6 mixed English/Russian sentences | automated call, 2026-11-13 |
| Chat channel | 1.5 s median answer time | same 20 questions |

## Unit costs (provider usage only)

Our main variable cost is ElevenLabs (speech recognition and voice). Gemini (the assistant's brain and the
chat) and Google Translate are small by comparison.

| Use | ElevenLabs usage we measured | Notes |
| --- | --- | --- |
| One short AI voice call (one question, one answer) | ~50 credits | 524 credits for 10 calls |
| Translated call between two people | ~2 credits per second of conversation (~225 credits for 110 s, 11 sentences) | voice output only; speech recognition is billed separately |
| Chat message | 0 ElevenLabs credits | Gemini only |

To turn credits into money, use the current ElevenLabs plan price (the Creator plan was about
$22/month for 100,000 credits when we started; check elevenlabs.io/pricing). Speech recognition time
is not yet broken out per call; measure it before quoting a per-minute price.

## Making the voice sound human

Today every translated sentence is spoken by one stock voice ("Sarah") on the fastest model
(Flash v2.5), whoever is speaking. Testers called it robotic. We re-voiced real call lines with
six models and four voices (including a clone of a team member's own voice); time to first audio:

| Model | Starts speaking after | Cost per character | Usable for live calls |
| --- | --- | --- | --- |
| Flash v2.5 (current) | 0.29 s | 0.5 credit | yes |
| v3 Conversational | 0.33–0.34 s | 0.5 credit | yes (most promising swap) |
| Multilingual v2 | 0.34–0.37 s | 1 credit | yes, twice the cost |
| v4 Turbo | 0.43–0.46 s | 0.5 credit | yes |
| v4 | 0.97 s | 1 credit | too slow |
| v3 | 1.10 s | 1 credit | too slow |

A cloned voice starts as fast as a stock voice (0.31 s on Flash) and costs the same per character.
The product direction: each person can speak in **their own cloned voice**, with consent; an officer
records a sample once, a student optionally before the call, and student clones are deleted after it.
Our plan allows 30 custom voices at a time.

## Speech-to-speech translation services (market check, third-party figures)

| Service | What it is | Speed | Price | Fit for us |
| --- | --- | --- | --- | --- |
| **Our pipeline** (speech recognition → translation → voice) | built, in use | **0.84 s** measured | see unit costs | keeps transcripts and full control; voice is a chosen or cloned voice |
| **Google Gemini 3.5 Live Translate** (June 2026, public preview) | direct speech-to-speech, keeps the speaker's intonation and pitch, 70+ languages | ~2.9 s median to first audio in one independent test (drift up to 14 s seen) | ~$0.04 per minute per direction reported (~$0.07/min for a two-way call); not confirmed on Google's price page | good for one-way talks (lectures); too slow for back-and-forth; no text transcript |
| **ElevenLabs Realtime Translation** (Oct 2026, alpha) | live translation into a chosen or cloned voice | not published | not public | enterprise workspaces only; not available on our plan |
| ElevenLabs Dubbing | keeps voice and emotion | file-based, not live | per minute | recordings and video, not calls |

Sources (third-party, check before quoting): Google blog "Fluid, natural voice translation with Gemini
3.5 Live Translate"; LiveLingo's Gemini 3.5 Live Translate guide (latency test); Orcarouter's Gemini 3.5
Live Translate comparison (pricing, two-way billing); Slator on ElevenLabs' live speech translation API;
ElevenLabs changelog 2026-10-19.

## Data and privacy notes for the plan

- The Gemini **free** tier may use requests to improve Google's products; real student calls need the
  paid tier (billing enabled on the Google Cloud project).
- Voice clones need the speaker's explicit consent; student clones should be deleted after the call.
- Call transcripts are stored by the Platform (per organisation); retention rules are not set yet.
