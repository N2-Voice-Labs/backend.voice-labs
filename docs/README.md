# Backend docs

> **UNAI** (formerly Voice Labs) is a B2B AI communication platform: AI voice agents (primary), real-time translation, AI chatbots and developer APIs. First market: university admissions offices. First MVP: the AI Admissions Assistant for international students at Inha University. Calls run in the browser over WebRTC (LiveKit); telephone/SIP calling is deferred.

Docs that describe the Platform, Runtime and Evaluation code live here, next to the code, so a
change and its documentation go in the same commit. Setup and onboarding (run locally,
quickstarts, troubleshooting) live in the `stack.voice-labs` docs site. Cleaned up 2026-11-11:
old plans, studies and superseded copies were deleted (they are in git history).

| Page | What it is |
| --- | --- |
| [Goal](goal.md) | UNAI product goal, Inha MVP journey and next proofs |
| [Status](status.md) | What works, at three evidence levels: implemented, mock-tested, real-provider-verified |
| [Contracts](contracts.md) | API and event contracts: authentication, sessions, translations, providers, events |
| [Smoke test](smoke-test.md) | Billable real-provider checklist and how to record results |
| [Calls](calls.md) | Reference: translated calls by shareable link |
| [Dubbing](dubbing.md) | Reference: the cinematic dubbing prototype, off by default |

## Conventions

- One topic, one page. Do not copy status, contracts or plans into another repository; link here.
- Put a date and a status line at the top of every plan; update both when you change it. When a
  plan is finished or abandoned, delete it instead of keeping it as history.
- Test counts and migration numbers go stale fast: link to the command or directory instead
  of copying a number, or date the number.
- A contract page changes in the same commit as the route it describes.
- Say what evidence a claim has: implemented, mock-tested, or real-provider-verified (see the
  legend in [Status](status.md)). Do not write "works" for something only a mock has seen.
