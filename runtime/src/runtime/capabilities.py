"""Provider capability metadata used during runtime selection."""

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
