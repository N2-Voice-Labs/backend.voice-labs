# Goal: an AI voice agent people can actually call

Status: **draft, 2026-11-11. Confirm with the team, then remove this line.** This page replaces
the earlier translator goal and the North Star plan (both are in git history). When it changes,
update the date.

## The product

A caller talks, by voice, to an AI agent that has a prompt, tools and knowledge documents. The
agent answers out loud, uses its tools during the call, and hands over to a human when it
should. Every call leaves a transcript, an outcome and latency numbers in Platform.

Voice Labs is the orchestration layer, not a model company. We do not build our own WebRTC,
SIP, STT, LLM, TTS or VAD. We connect existing ones (LiveKit, Pipecat, Deepgram or ElevenLabs,
OpenAI or Gemini, Cartesia or ElevenLabs, Silero), make them swappable through the provider
registries, route the audio correctly, measure every turn, and make the whole thing reliable.

## Where it stands (2026-11-11)

Evidence levels are those in [Status](status.md): ✅ real providers, 🧪 mocks only.

| Part of an agent call | Evidence |
| --- | --- |
| Browser call: speech → LLM → speech | ✅ October 2026, about 1.65 s from end of speech to first reply audio. Not re-run on today's `main`. |
| Agent + translator (English agent, Russian caller) | ✅ October 2026, n=2, about 1.9 s |
| Prompt, language, model, voice settings; templates | ✅ |
| Transcripts, outcomes, signed events | ✅ |
| Tools (HTTP and built-in `record_outcome`, `transfer_to_human`, `search_knowledge`) | 🧪 |
| Knowledge documents (keyword search) | 🧪 |
| Interruption (barge-in) | 🧪, time to stop never measured |
| Handoff to a human | 🧪 |
| Deployed somewhere a stranger can use it | ❌ |

## The next proofs, in order

Each one is done only when it has been run with real providers and the result is written into
[Status](status.md) with a date and numbers.

1. **Re-run the browser agent call on today's `main`.** A lot changed since October (provider
   registry, ElevenLabs, Google/Gemini, routing). Record p50 and p95 reply latency over at least
   20 turns.
2. **Tools and knowledge with real providers.** One agent with one HTTP tool and one knowledge
   document answers questions that need both. Record whether the right tool was called and the
   answer was correct.
3. **Interruption, measured.** Time from the caller starting to speak to the agent's audio
   stopping, p50 and p95.
4. **Handoff to a human.** In the browser first (a person joins the room); phone transfer waits
   for a SIP trunk.
5. **A link a stranger can use.** Platform and Runtime deployed, one agent behind a public page,
   10 calls in a row without a failure.

## Not now

These exist in code or in plans, but they are not the goal and do not get work until the
proofs above are done:

- Two-person translated calls and cinematic dubbing ([Calls](calls.md), [Dubbing](dubbing.md)
  describe the code that exists; keep it working, do not extend it).
- Phone (SIP) calls and campaigns: blocked on a SIP account and number anyway.
- Messenger channels, text chat, voice cloning, live self-cloning.
- New languages (Kazakh, Uzbek, Kyrgyz) and mixed-language speech: no supported speech
  provider is wired in.
