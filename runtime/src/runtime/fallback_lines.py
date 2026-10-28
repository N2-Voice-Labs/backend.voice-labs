"""Short spoken fallback lines (ST-12) in the caller's language.

``line(kind, language)`` returns the line for ``kind`` in ``language``; an unknown or
unsupported language falls back to English, so the caller always hears something.
"""

from __future__ import annotations

REPEAT = "repeat"
CHECKING = "checking"
IDLE = "idle"

FALLBACK_LANGUAGE = "en"

_LINES: dict[str, dict[str, str]] = {
    REPEAT: {
        "en": "Sorry, could you say that again?",
        "ru": "Извините, не могли бы вы повторить это?",
        "ko": "죄송하지만 다시 한 번 말씀해 주시겠어요?",
        "es": "Perdón, ¿podrías repetirlo?",
        "fr": "Pardon, pourriez-vous répéter ?",
        "de": "Entschuldigung, könnten Sie das noch einmal sagen?",
        "ja": "すみません、もう一度おっしゃっていただけますか？",
        "zh": "抱歉，您能再说一遍吗？",
        "pt": "Desculpe, poderia repetir?",
        "it": "Scusi, potrebbe ripetere?",
        "nl": "Sorry, kunt u dat nog een keer zeggen?",
        "hi": "माफ़ कीजिए, क्या आप वह दोबारा कह सकते हैं?",
        "tr": "Üzgünüm, bunu tekrar söyler misiniz?",
        "pl": "Przepraszam, czy możesz to powtórzyć?",
    },
    CHECKING: {
        "en": "Let me check that again…",
        "ru": "Давайте я проверю это ещё раз…",
        "ko": "다시 한 번 확인해 볼게요…",
        "es": "Déjame verificarlo de nuevo…",
        "fr": "Laissez-moi vérifier à nouveau…",
        "de": "Ich prüfe das noch einmal…",
        "ja": "もう一度確認しますね…",
        "zh": "我再核实一下…",
        "pt": "Deixe-me verificar isso novamente…",
        "it": "Fammi controllare di nuovo…",
        "nl": "Ik controleer dat nog even…",
        "hi": "मैं इसे दोबारा जाँचता हूँ…",
        "tr": "Bunu bir daha kontrol edeyim…",
        "pl": "Sprawdzę to jeszcze raz…",
    },
    IDLE: {
        "en": "Are you still there?",
        "ru": "Вы всё ещё здесь?",
        "ko": "아직 계신가요?",
        "es": "¿Sigues ahí?",
        "fr": "Vous êtes toujours là ?",
        "de": "Sind Sie noch da?",
        "ja": "まだいらっしゃいますか？",
        "zh": "您还在吗？",
        "pt": "Você ainda está aí?",
        "it": "È ancora lì?",
        "nl": "Bent u er nog?",
        "hi": "क्या आप अभी भी वहाँ हैं?",
        "tr": "Hâlâ orada mısınız?",
        "pl": "Czy nadal tam jesteś?",
    },
}


def _primary_subtag(language: str | None) -> str:
    """``"pt-BR"`` / ``"ru_RU"`` / ``"EN"`` -> ``"pt"`` / ``"ru"`` / ``"en"``."""

    if not language:
        return FALLBACK_LANGUAGE
    return language.strip().replace("_", "-").split("-", 1)[0].lower()


def line(kind: str, language: str | None) -> str:
    """Return the fallback line of ``kind`` for ``language``, English when unsupported."""

    try:
        lines = _LINES[kind]
    except KeyError:
        raise ValueError(f"unknown fallback line kind: {kind!r}") from None
    return lines.get(_primary_subtag(language), lines[FALLBACK_LANGUAGE])
