# evals.voice-labs

The package currently provides deterministic generic WER and reference-conditioned
terminology recall metrics. It does not yet include a dataset schema, code-switch-specific
normalization, benchmark runner, or latency report. The unit tests do not measure model
quality. Private recordings are intentionally not included.

Future benchmark records should capture provider/model versions, dataset version, hardware,
WER by language and code-switch condition, terminology recall, and measured latency stages.
