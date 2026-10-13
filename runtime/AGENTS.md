# Runtime agent instructions

This repository owns live RuntimeSession state, transport, pipeline, provider selection,
cancellation, and versioned events. Durable business state belongs to platform.voice-labs.
Prefer native Pipecat and LiveKit integrations; do not recreate media, VAD, or inference
infrastructure. Run `python -m pytest -q`. Voice extras are provider-dependent and must
not be reported as working unless exercised locally.
