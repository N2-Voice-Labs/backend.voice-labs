# Backend docs

Docs that describe the Platform, Runtime and Evaluation code live here, next to the code, so
a change and its documentation go in the same commit. Last reviewed 2026-11-05.

**Current** (describes the code; update with the code):

| Page | What it is |
| --- | --- |
| [Contracts](contracts.md) | API and event contracts: authentication, translation sessions (`/translations`), ElevenLabs (live TTS), voices, interpreter mode, events |
| [Status](status.md) | What works, with three evidence levels: implemented, mock-tested, real-provider-verified; fresh test results |
| [Smoke test](smoke-test.md) | Billable real-provider checklist. The automated ElevenLabs-layer script ran on 2026-11-05 (15 of 15 passed; results recorded); the manual two-person parts are pending. Includes how to give the script a key and rotate it, and a keyless PostgreSQL check recipe (verified 2026-11-05) |
| [Stale stack docs](stale-stack-docs.md) | Pages in `stack.voice-labs/docs` that are out of date, with paste-ready replacements |
| [Stack acceptance patch](stack-acceptance-patch.md) | Three-line fix for the stale stack acceptance test (not applied; sibling repo) |

**Plans** (direction and task history; status lines are dated, old entries are kept as history):

| Page | What it is |
| --- | --- |
| [North Star](north-star.md) | The goal, the first proof, the phased plan |
| [Translate plan](translate-plan.md) | Interpreter milestone and task history |

`goal.md` is the task specification for the current renovation work, not a maintained page; it
is not part of this index and is not edited with the other docs.

## Where everything else is

These were copied (not moved) from `stack.voice-labs/docs`, which is a separate repository and
was not edited. The maintained copies are Contracts, Status, North Star and Translate plan above;
the stack versions of them are stale (paste-ready pointer pages are in
[Stale stack docs](stale-stack-docs.md)).

Still only in `stack.voice-labs/docs` (infrastructure and onboarding, or historical): run
locally, quickstarts, troubleshooting, development, evaluation, concepts, architecture and
C4 diagrams, the MVP/tasks/studio/live/roadmap plans, the project report and overview.
Several of those describe the first agent-only slice and the older "no organizations" scope
(for example `BACKEND_ARCHITECTURE.md`, `BACKEND_TASKS.md`, `index.md`); treat them as history
until reviewed.

## Conventions

- Put a date and a status line at the top of every plan; update both when you change it.
- Test counts and migration numbers go stale fast: link to the command or directory instead
  of copying a number, or date the number.
- A contract page changes in the same commit as the route it describes.
- Say what evidence a claim has: implemented, mock-tested, or real-provider-verified (see the
  legend in [Status](status.md)). Do not write "works" for something only a mock has seen.
- Dated benchmark tables are history; do not read them as numbers for the current code.
