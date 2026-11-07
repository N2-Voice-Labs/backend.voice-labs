Our next objective is to determine whether cinematic dubbing actually improves the experience of real-time translated conversations.

The technical prototype is complete. Now we need to establish whether it delivers the intended product value.

### Primary goal

Make translated calls feel like natural communication between two real humans rather than interaction with an AI-generated voice.

We want to preserve the original speaker's identity, emotion, and human presence while ensuring the listener clearly understands the translation.

The critical question is whether keeping the original voice audible improves trust and naturalness without making conversations confusing, distracting, or slower.

### What I want to achieve

Create realistic audio demonstrations of different listening experiences using our existing dubbing system.

Compare:

**AI-only translation:** The listener hears only the translated voice.

**Current cinematic dubbing:** The original speaker is heard with a slight delay, and their volume is dynamically reduced when translation begins.

**Alternative cinematic dubbing:** Explore whether replaying a delayed portion of the original speech underneath the translation creates a more natural documentary-style experience.

The third option is exploratory. Do not implement a new production architecture unless the listening results justify it.

### Key considerations

Our current implementation has an important limitation: during isolated sentences, the original voice finishes before the translation starts, so there is no meaningful overlap.

Investigate whether this actually matters for perceived naturalness.

Avoid sacrificing the existing low translation latency merely to create synchronized audio.

Pay particular attention to:

- How natural the transitions between original and translated voices sound.
- Whether the original voice makes the call feel more human.
- Whether simultaneous languages interfere with comprehension.
- Whether the original voice should be heard continuously or only briefly.
- How the experience behaves during interruptions and overlapping conversation.
- Whether users might mistake the translated voice for an AI caller rather than a real person using translation.

### Validation approach

Use existing recordings or synthetic audio to generate realistic listening examples.

Include short utterances, longer sentences, continuous speech, natural pauses, and interruptions.

Make the examples easy for a human listener to compare.

Preserve the same source content and translation across versions so the comparison isolates the audio presentation.

Recommend an objective listening evaluation that measures trust, intelligibility, naturalness, and preference.

Do not claim that the experience is better based solely on audio metrics. Human listening is required.

### Technical constraints

Keep production dubbing disabled by default.

Do not modify the existing translation architecture, STT, TTS, fallback, or latency settings.

Do not consume additional ElevenLabs credits.

Use existing audio and offline processing wherever possible.

Investigate echo risk and resource consumption, but prioritize evaluating the actual listening experience over further engineering optimization.

### Definition of success

We should finish with realistic, directly comparable audio examples and a clear understanding of what remains to be validated by human listeners.

The final deliverable should include:

1. Playable audio comparisons of the different approaches.
2. An explanation of how each approach changes the listening experience.
3. A recommended configuration for an initial human listening test.
4. A simple evaluation protocol with measurable criteria.
5. Technical limitations that could affect real phone calls.
6. A recommendation on which experience should become our default if listener testing supports it.

Do not prematurely declare cinematic dubbing better than AI-only translation.

The goal is to discover the best human-to-human communication experience, not simply demonstrate that our mixer works.
