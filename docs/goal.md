# UNAI goal: AI Admissions Assistant for Inha University

Status: **current product direction, 2026-11-12.** MVP implementation is in progress.

> **UNAI** (formerly Voice Labs) is a B2B AI communication platform: AI voice agents (primary), real-time translation, AI chatbots and developer APIs. First market: university admissions offices. First MVP: the AI Admissions Assistant for international students at Inha University. Calls run in the browser over WebRTC (LiveKit); telephone/SIP calling is deferred.

## Product and MVP

UNAI is Voice Labs renamed and repositioned, not a separate company. It is an AI-as-a-Service platform designed for multiple industries. AI voice agents are the primary product; real-time translation supports a handoff when a caller and staff member speak different languages. AI chatbots are a secondary channel using the same knowledge. Developer APIs are the commercial direction; today this means the existing Platform HTTP API. Billing, API-key portal and developer dashboard are planned, not built.

The first demo is for Inha University international admissions. The platform remains industry-neutral: university-specific knowledge and prompts belong in agent data, not runtime code.

MVP journey: a student opens the Inha portal and starts an AI voice call in the browser. The agent answers from verified admissions information. On “Connect to Admissions Officer,” staff join from a browser `/staff` console; the student can speak English while the officer speaks Korean through the existing `/calls` interpreter and private listen rooms. Either side can end the call. A text chatbot is an alternative channel. No phone number or SIM is needed. Traditional telephone/SIP calling and outbound campaigns are deferred.

Small knowledge bases are intended to be inserted inline into the agent prompt; larger ones use the existing keyword `search_knowledge` tool. Inline mode is in progress; embeddings RAG is deferred.

## Existing evidence

Evidence below records prior runs and remains historical; it does not establish that the current MVP is complete. See [Status](status.md) for evidence levels.

| Capability | Recorded evidence |
| --- | --- |
| Browser speech → LLM → speech | ✅ October 2026, about 1.65 s from end of speech to first reply audio; not re-run on current code. |
| Agent + translator (English agent, Russian caller) | ✅ October 2026, n=2, about 1.9 s; historical run. |
| Agent settings, templates, transcripts and signed events | ✅ previously verified as recorded in Status. |
| Tools and keyword knowledge search | 🧪 mocks only. |
| Interruption and handoff | 🧪 mocks only; no human browser handoff. |

## Next proofs, in order

Record dated outcomes in [Status](status.md). MVP pieces remain in progress until each path has real evidence.

1. **Public demo voice session:** start and end a student browser session with the intended demo agent; verify provider-backed calls and safe session cleanup.
2. **Inha admissions knowledge:** load verified admissions facts into the agent, confirm inline small-KB behavior or keyword search for larger content, and evaluate the FAQ set.
3. **Human handoff:** student requests an officer, staff see and accept the handoff in `/staff`, and both join the browser call.
4. **Translated handoff:** verify English↔Korean speech in both directions through private listen rooms.
5. **Chat alternative:** answer from the same agent knowledge and verify grounding.
6. **Demo readiness:** run the acceptance targets below and record evidence; deploy the public demo only when its operating and privacy details are ready.

## Acceptance targets (not yet measured)

Every value below is a **target, not yet measured**.

| Acceptance measure | Target | Status |
| --- | --- | --- |
| Consecutive AI calls | 10 without failure | Target, not yet measured |
| Verified Inha admissions FAQ answers | ≥18/20 | Target, not yet measured |
| Median voice reply latency | <3 s after end of speech | Target, not yet measured |
| Devices connected to one handoff call | 2 | Target, not yet measured |
| Translated exchanges | 5 each direction (English→Korean and Korean→English) | Target, not yet measured |
| Clean session terminations | 10 | Target, not yet measured |
| Real mobile browser | ≥1 successful run | Target, not yet measured |
| Demo duration | 3 minutes | Target, not yet measured |

## Deferred

Telephone/SIP deployment, phone numbers, SIMs and outbound campaigns; embeddings RAG; billing and developer portal/dashboard; additional vertical-specific runtime logic. Existing SIP code paths are documented as technical reference, not part of the MVP path.
