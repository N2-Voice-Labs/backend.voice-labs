"""Provider capability metadata used during runtime selection.

Each provider declares its features when it registers in ``speech_providers``. Known
features: ``streaming``; stt ``keyterms`` (vocabulary boosting) and ``multi_language``
(per-utterance language detection, the ``multi`` language); llm ``tools``; tts
``voice_clone`` and ``transport_destination`` (per-listener output tracks).
"""

from dataclasses import dataclass
from typing import FrozenSet, Literal

ProviderKind = Literal["stt", "llm", "tts"]


@dataclass(frozen=True)
class ProviderCapabilities:
    kind: ProviderKind
    provider: str
    features: FrozenSet[str]

    def supports(self, feature: str) -> bool:
        return feature in self.features
