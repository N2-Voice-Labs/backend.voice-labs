Our next objective is to develop a natural, cinematic real-time dubbing experience for human-to-human translated calls.

The fundamental product goal is to make users feel that they are communicating with a real person through a translator, rather than speaking to an AI assistant.

Currently, the listener hears only the AI-translated voice. We want to investigate and implement an alternative experience where the original speaker's voice remains present, creating a documentary or movie-style dubbing effect.

### Desired experience

When someone speaks:

- The listener initially hears the real speaker's voice.
- The original audio may be delayed slightly to create a more natural transition into the translation.
- When the AI translation begins, it becomes the dominant voice.
- The original voice fades into the background instead of abruptly disappearing.
- The listener can still perceive the original speaker's emotion, intonation, and human presence.
- The translated speech remains clear and understandable.

The intended result is natural, trustworthy human-to-human communication.

### Primary requirements

**1. Preserve latency**

The existing translation pipeline must remain unaffected.

Do not delay translated speech to synchronize it with the original audio. The original audio should adapt to the translation, not the opposite way around.

**2. High-quality audio mixing**

Investigate and implement a suitable approach to delaying, buffering, mixing, and dynamically ducking the original speaker's audio.

Experiment with different delay and volume configurations to identify which creates the most natural dubbing experience.

Avoid distracting overlaps, abrupt volume changes, clipping, echo, and unnecessary audio artifacts.

**3. Correct audio routing**

The listener should hear the appropriate combination of the remote speaker's original voice and translated speech.

The original speaker must not hear their own delayed voice.

Ensure the mixer does not interfere with STT processing, VAD, turn detection, or translation.

**4. Configurable experience**

Implement the feature behind a session-level setting that defaults to disabled.

The architecture should support switching between translation-only and cinematic dubbing modes without rebuilding the translation pipeline.

**5. Maintain production reliability**

Preserve existing translation performance, fallback behavior, concurrency isolation, and connection management.

Avoid introducing significant per-call CPU or memory overhead.

Ensure buffers and other resources are released when calls end or fail.

### Development approach

You have discretion over the technical implementation.

First, examine the existing LiveKit audio architecture and identify the cleanest integration point.

Then implement the smallest maintainable solution that achieves the intended experience.

Validate it through automated tests, offline audio mixing experiments, and latency measurements.

Use existing audio recordings or synthetic test data where possible.

Do not consume ElevenLabs credits or change the default production experience.

### Evaluation criteria

Evaluate the prototype against our existing AI-only experience using:

- Audio intelligibility.
- Naturalness of voice transitions.
- Preservation of the speaker's human presence.
- Added end-to-end latency.
- Audio artifacts and echo risk.
- CPU and memory overhead.
- Reliability during interruptions, short utterances, and overlapping speech.

Do not assume cinematic dubbing automatically increases trust. Treat that as a product hypothesis requiring actual listener testing.

### Definition of success

We should have a working, configurable dubbing prototype that preserves the existing translation pipeline's performance and allows us to compare different audio experiences.

The implementation should be production-safe when disabled, independently testable, and ready for a future listening study.

Run relevant automated tests and commit verified changes.

When finished, report:

- What was implemented.
- How the mixing and timing work.
- What the resulting audio experience is like.
- Any measurable latency or resource overhead.
- Limitations and unresolved issues.
- Whether this is ready for live user testing.

Prioritize the quality of the actual conversation over adding unnecessary features.
