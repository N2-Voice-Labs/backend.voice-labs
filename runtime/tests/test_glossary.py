"""NS-08 glossary: parsing, prompt injection, keyterms and the translators that use it."""

import asyncio

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from src.runtime.agent_translation import (
    AgentTranslationSettings,
    CallerLanguageState,
    CallerLanguageTap,
    InboundTranslator,
    OutboundTranslator,
)
from src.runtime.glossary import (
    MAX_PROMPT_ENTRIES,
    SEND_ALL_THRESHOLD,
    glossary_keyterms,
    glossary_prompt,
    parse_glossary,
    select_entries,
)
from src.runtime.interpreter import (
    InterpreterSettings,
    TranslationProcessor,
    build_translation_messages,
)

RAW = [
    {
        "source": "residence permit",
        "translations": {"ru": "вид на жительство", "ko": "체류 허가"},
        "note": "immigration document",
    },
    {"source": "Voice Labs", "keep": True},
]
GLOSSARY = parse_glossary(RAW)


def system_prompt(text, source, target, glossary=GLOSSARY, **kwargs) -> str:
    return build_translation_messages(text, source, target, glossary=glossary, **kwargs)[0][
        "content"
    ]


# -- parsing ---------------------------------------------------------------------


def test_parse_glossary() -> None:
    assert parse_glossary(None) == () and parse_glossary([]) == ()
    permit, brand = GLOSSARY
    assert permit.source == "residence permit" and permit.note == "immigration document"
    assert permit.rendering("ru") == "вид на жительство"
    assert permit.rendering("en") == "residence permit"  # the language without an entry
    assert brand.keep and brand.rendering("ru") == "Voice Labs"


@pytest.mark.parametrize(
    "bad",
    [
        "x",
        [None],
        [{"translations": {"ru": "x"}}],
        [{"source": "x"}],
        [{"source": "x", "translations": {"ru": ""}}],
        [{"source": "x", "translations": []}],
        [{"source": "x", "keep": "yes"}],
        [{"source": "x", "keep": True, "note": 3}],
    ],
)
def test_parse_glossary_rejects_malformed(bad) -> None:
    with pytest.raises(RuntimeError):
        parse_glossary(bad)


def test_settings_carry_the_glossary() -> None:
    interpreter = InterpreterSettings.from_mapping(
        {"languages": ["en", "ru"], "direction": "auto", "glossary": RAW}
    )
    assert interpreter.glossary == GLOSSARY
    assert InterpreterSettings.from_mapping({"languages": ["en", "ru"]}).glossary == ()
    translation = AgentTranslationSettings.from_mapping(
        {"working_language": "en", "caller_languages": ["ru"], "glossary": RAW}
    )
    assert translation.glossary == GLOSSARY
    with pytest.raises(RuntimeError):
        InterpreterSettings.from_mapping({"languages": ["en", "ru"], "glossary": [{}]})


# -- prompt ---------------------------------------------------------------------


def test_residence_permit_gets_the_required_russian_term() -> None:
    prompt = system_prompt("I need a residence permit.", "en", "ru")
    assert '"residence permit" => "вид на жительство"' in prompt
    assert "(immigration document)" in prompt
    assert 'Voice Labs"' in prompt and "Never translate" in prompt


def test_glossary_works_in_the_reverse_direction_and_for_other_languages() -> None:
    assert '"вид на жительство" => "residence permit"' in system_prompt(
        "Мне нужен вид на жительство", "ru", "en"
    )
    assert '"residence permit" => "체류 허가"' in system_prompt("residence permit", "en", "ko")
    assert '"체류 허가" => "residence permit"' in system_prompt("체류 허가", "ko", "en")
    # ru <-> ko: both sides have an entry.
    assert '"вид на жительство" => "체류 허가"' in system_prompt("вид на жительство", "ru", "ko")


def test_unrelated_language_pair_gets_only_keep_terms() -> None:
    prompt = system_prompt("Hola", "en", "es")
    assert "Glossary" not in prompt and "required translation" not in prompt
    assert '"Voice Labs"' in prompt


def test_no_glossary_leaves_the_prompt_unchanged() -> None:
    plain = build_translation_messages("Hi", "en", "ru")
    assert build_translation_messages("Hi", "en", "ru", glossary=()) == plain
    assert "Glossary" not in plain[0]["content"] and "Never translate" not in plain[0]["content"]


def test_glossary_comes_before_history_and_user_text_is_untouched() -> None:
    history = [("en", "Hello", "ru", "Привет")]
    messages = build_translation_messages(
        "residence permit", "en", "ru", history, glossary=GLOSSARY
    )
    system = messages[0]["content"]
    assert system.index("Glossary") < system.index("Earlier utterances")
    assert messages[1] == {"role": "user", "content": "residence permit"}


def test_terms_cannot_break_out_of_the_prompt_lines() -> None:
    glossary = parse_glossary([{"source": 'a"b', "translations": {"ru": 'c"d'}}])
    prompt = glossary_prompt(glossary, "a\"b", "en", "ru")
    assert r'"a\"b" => "c\"d"' in prompt


def name(n: int) -> str:
    """A distinct alphabetic term (digits would make the inflection stems collide)."""

    return f"{chr(97 + n // 26)}{chr(97 + n % 26)}xyzq"


def make_big(count):
    return parse_glossary(
        [
            {"source": name(n), "translations": {"ru": name(n).replace("q", "щ")}}
            for n in range(count)
        ]
    )


def test_small_glossary_is_sent_whole_and_large_is_filtered_by_relevance() -> None:
    small = make_big(SEND_ALL_THRESHOLD)
    assert len(select_entries(small, "unrelated words", "en", "ru")) == SEND_ALL_THRESHOLD

    large = make_big(150)
    assert select_entries(large, "unrelated words", "en", "ru") == []
    chosen = select_entries(large, f"see {name(7).upper()} and {name(120)} please", "en", "ru")
    assert [e.source for e in chosen] == [name(7), name(120)]
    # Relevance works on the source side in the other direction too.
    ru = name(42).replace("q", "щ")
    assert [e.source for e in select_entries(large, f"Нужен {ru}", "ru", "en")] == [name(42)]


def test_prompt_cap() -> None:
    large = make_big(150)
    text = " ".join(name(n) for n in range(150))
    assert len(select_entries(large, text, "en", "ru")) == MAX_PROMPT_ENTRIES
    prompt = glossary_prompt(large, text, "en", "ru")
    assert prompt.count("=>") == MAX_PROMPT_ENTRIES


def test_inflected_forms_match_in_large_glossaries() -> None:
    glossary = parse_glossary(
        [{"source": "residence permit", "translations": {"ru": "вид на жительство"}}]
        + [{"source": name(n), "translations": {"ru": name(n)}} for n in range(20)]
    )
    chosen = select_entries(glossary, "Покажите мой вид на жительства", "ru", "en")
    assert [e.source for e in chosen] == ["residence permit"]
    assert [e.source for e in select_entries(glossary, "two residence permits", "en", "ru")] == [
        "residence permit"
    ]


# -- keyterms -------------------------------------------------------------------


def test_glossary_keyterms() -> None:
    assert glossary_keyterms(GLOSSARY) == [
        "residence permit",
        "вид на жительство",
        "체류 허가",
        "Voice Labs",
    ]
    assert glossary_keyterms(GLOSSARY, "ru") == ["вид на жительство", "Voice Labs"]
    assert glossary_keyterms(GLOSSARY, "en") == ["residence permit", "Voice Labs"]
    settings = InterpreterSettings(languages=("en", "ru"), glossary=GLOSSARY)
    assert glossary_keyterms(settings, "ru") == ["вид на жительство", "Voice Labs"]
    assert glossary_keyterms(()) == []
    big = make_big(300)
    assert len(glossary_keyterms(big)) == 100


# -- processors -------------------------------------------------------------------


def run(processor, frames):
    return asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))


def test_interpreter_processor_sends_the_glossary_to_the_translator() -> None:
    seen = []

    async def translate(messages):
        seen.append(messages)
        return "Мне нужен вид на жительство"

    processor = TranslationProcessor(
        settings=InterpreterSettings(languages=("en", "ru"), glossary=GLOSSARY),
        translate=translate,
        model="m",
        initial_tts_language="ru",
    )
    run(
        processor,
        [
            UserStartedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            TranscriptionFrame(
                "I need a residence permit.", "speaker", "2026-01-01T00:00:00Z", Language.EN, finalized=True
            ),
            SleepFrame(sleep=0.02),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ],
    )
    (messages,) = seen
    assert '"residence permit" => "вид на жительство"' in messages[0]["content"]


def test_agent_translators_send_the_glossary() -> None:
    seen = []

    async def translate(messages):
        seen.append(messages)
        return "ok"

    settings = AgentTranslationSettings(
        working_language="en", caller_languages=("ru",), glossary=GLOSSARY
    )
    state = CallerLanguageState(settings)
    state.language = "ru"
    inbound = Pipeline(
        [CallerLanguageTap(state), InboundTranslator(state=state, translate=translate, model="m")]
    )
    context = LLMContext(messages=[{"role": "user", "content": "Нужен вид на жительство"}])
    run(
        inbound,
        [
            TranscriptionFrame("Нужен вид на жительство", "caller", "t", Language.RU, finalized=True),
            SleepFrame(sleep=0.02),
            LLMContextFrame(context=context),
            SleepFrame(sleep=0.05),
        ],
    )
    (inbound_messages,) = seen
    assert '"вид на жительство" => "residence permit"' in inbound_messages[0]["content"]

    seen.clear()
    outbound = OutboundTranslator(state=state, translate=translate, model="m")
    run(
        outbound,
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("You can apply for a residence permit."),
            LLMFullResponseEndFrame(),
        ],
    )
    (outbound_messages,) = seen
    assert '"residence permit" => "вид на жительство"' in outbound_messages[0]["content"]


def test_agent_instruction_protects_keep_terms() -> None:
    settings = AgentTranslationSettings(
        working_language="en", caller_languages=("ru",), glossary=GLOSSARY
    )
    assert "Never alter these names or terms: Voice Labs." in settings.instruction
    plain = AgentTranslationSettings(working_language="en", caller_languages=("ru",))
    assert "Never alter" not in plain.instruction
