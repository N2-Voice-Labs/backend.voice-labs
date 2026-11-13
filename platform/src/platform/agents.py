"""Agent and provider-reference routes, with the agent config validators."""

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from sqlalchemy.orm import Session

from sqlalchemy import select

from .context import AuthContext, require
from .models import Agent as AgentRecord
from .models import ProviderConfig, Voice
from .tenancy import bypass, get_owned, get_public_agent, get_scoped_db, scoped
from .validation import (
    AUTO_DIRECTION_LANGUAGES,
    DEFAULT_LATENCY_MODE,
    INTERPRETER_LANGUAGES,
    LATENCY_MODES,
    REALTIME_TRANSLATE_OUTPUT_LANGUAGES,
    reject_secret_configuration,
    validate_enabled_languages,
    validate_glossary,
)

router = APIRouter()


class InterpreterConfig(BaseModel):
    languages: list[str]
    direction: str = "auto"
    speaker_language: Literal["declared", "auto"] = "declared"
    engine: Literal["cascaded", "realtime"] = "cascaded"
    # RM-011: "instant" | "balanced" | "accurate". Checked in ``validate_interpreter_engine``
    # (422 invalid_interpreter_config). ``instant`` picks the realtime engine when the
    # config allows it and ``engine`` is not given; see ``resolve_interpreter_engine``.
    latency_mode: str = DEFAULT_LATENCY_MODE
    # RM-002: two-person calls also hear each other's original microphone (off by default).
    hear_original: bool = False
    # Cinematic dubbing (Runtime ``dubbing``): the other person's real voice, quiet and ducked,
    # under the translation. Off (absent from the stored config) unless set; replaces
    # ``hear_original``. true = the default preset, or a preset name.
    dubbing: bool | Literal["subtle", "balanced", "cinematic", "present"] | None = None
    # NS-08: terms the translation must respect; validated and normalized by
    # ``validate_glossary``. Absent from the stored config when not given.
    glossary: list[dict[str, Any]] | None = None

    @field_validator("glossary", mode="before")
    @classmethod
    def validate_glossary_field(cls, value: Any) -> list[dict[str, Any]] | None:
        return None if value is None else validate_glossary(value)

    @model_serializer(mode="wrap")
    def omit_empty_glossary(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if data.get("speaker_language") == "declared" and "speaker_language" not in self.model_fields_set:
            data.pop("speaker_language", None)
        if not data.get("glossary"):
            data.pop("glossary", None)
        if not data.get("dubbing"):  # off: stored and hashed exactly as before the option existed
            data.pop("dubbing", None)
        return data

    @model_validator(mode="after")
    def dubbing_replaces_hear_original(self) -> "InterpreterConfig":
        if self.dubbing and self.hear_original:
            raise ValueError("dubbing replaces hear_original: set only one")
        return self

    @field_validator("languages", mode="before")
    @classmethod
    def validate_languages(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("languages must be a list")
        if len(value) != 2 or len(set(value)) != 2:
            raise ValueError("languages must contain exactly 2 distinct items")
        for lang in value:
            if lang not in INTERPRETER_LANGUAGES:
                raise ValueError(f"unknown language: {lang}")
        return [lang.lower() for lang in value]

    @field_validator("direction", mode="before")
    @classmethod
    def validate_direction(cls, value: str) -> str:
        if value not in ("auto", "one_way"):
            raise ValueError("direction must be 'auto' or 'one_way'")
        return value

    @field_validator("direction", mode="after")
    @classmethod
    def validate_auto_direction_languages(cls, value: str, info) -> str:
        if value == "auto" and "languages" in info.data:
            languages = info.data["languages"]
            if not all(lang in AUTO_DIRECTION_LANGUAGES for lang in languages):
                raise ValueError(
                    "auto direction requires both languages in Deepgram's code-switching set"
                )
        return value


REALTIME_GLOSSARY_MESSAGE = (
    "Glossary is not supported by the realtime engine; use step-by-step (cascaded)"
)


def validate_interpreter_engine(interpreter: Any) -> None:
    """Reject engine/direction/language combinations the engine cannot serve (422).

    ``realtime`` (OpenAI gpt-realtime-translate) translates one audio stream into one
    fixed output language per session and does not report the spoken language, so it
    supports ``one_way`` only and its target (``languages[1]``) must be one of the
    model's output languages.
    """

    if interpreter is None:
        return
    if isinstance(interpreter, InterpreterConfig):
        interpreter = interpreter.model_dump()
    if interpreter.get("latency_mode", DEFAULT_LATENCY_MODE) not in LATENCY_MODES:
        raise HTTPException(status_code=422, detail="invalid_interpreter_config")
    if interpreter.get("engine", "cascaded") != "realtime":
        return
    if interpreter.get("glossary"):
        # ST-41: gpt-realtime-translate accepts no instructions or glossary.
        raise HTTPException(status_code=422, detail=REALTIME_GLOSSARY_MESSAGE)
    languages = interpreter.get("languages") or []
    if (
        interpreter.get("direction") != "one_way"
        or len(languages) != 2
        or languages[1] not in REALTIME_TRANSLATE_OUTPUT_LANGUAGES
    ):
        raise HTTPException(status_code=422, detail="invalid_interpreter_config")


def resolve_interpreter_engine(
    interpreter: dict[str, Any], engine_explicit: bool
) -> dict[str, Any]:
    """Apply the latency mode's engine choice (RM-011) to a validated interpreter dict.

    ``instant`` selects the ``realtime`` engine when none was given explicitly and the
    config allows it (``one_way`` and a target language the realtime model can speak);
    otherwise the engine stays as stored (``cascaded``, tuned by the mode in Runtime). A
    non-empty ``glossary`` keeps ``instant`` on the cascaded engine (ST-41): the realtime
    model cannot honour it. An explicit ``engine`` always wins. The stored engine is
    therefore what Runtime runs.
    Note: realtime serves one speaker, so a two-person session on an ``instant`` agent
    needs ``engine: "cascaded"`` set explicitly.
    """

    languages = interpreter.get("languages") or []
    if (
        not engine_explicit
        and not interpreter.get("glossary")
        and interpreter.get("latency_mode") == "instant"
        and interpreter.get("direction", "auto") == "one_way"
        and len(languages) == 2
        and languages[1] in REALTIME_TRANSLATE_OUTPUT_LANGUAGES
    ):
        return {**interpreter, "engine": "realtime"}
    return interpreter


MAX_CALLER_LANGUAGES = 4


def validate_agent_translation(
    mode: str, translation: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Validate an agent's ``translation`` block (TR-040) and return it normalized.

    ``{"working_language": code, "caller_languages": [codes]}``: the agent reasons in
    the working language while callers speak 1-4 other languages. Caller languages are
    detected per utterance only when every caller language and the working language are
    in Deepgram's code-switching set (``AUTO_DIRECTION_LANGUAGES``); otherwise exactly
    one (fixed) caller language is allowed. An optional ``glossary`` (NS-08, see
    ``validate_glossary``) is normalized and kept.

    Raises 422 ``translation_not_allowed_for_interpreter``, ``invalid_translation_config``
    or ``invalid_glossary: <reason>``.
    """

    if translation is None:
        return None
    if mode == "interpreter":
        raise HTTPException(
            status_code=422, detail="translation_not_allowed_for_interpreter"
        )
    invalid = HTTPException(status_code=422, detail="invalid_translation_config")
    if (
        not isinstance(translation, dict)
        or not {"working_language", "caller_languages"} <= set(translation)
        or set(translation) - {"working_language", "caller_languages", "glossary"}
    ):
        raise invalid
    working = translation["working_language"]
    callers = translation["caller_languages"]
    if (
        not isinstance(working, str)
        or not isinstance(callers, list)
        or not all(isinstance(code, str) for code in callers)
    ):
        raise invalid
    working = working.lower()
    callers = [code.lower() for code in callers]
    if (
        working not in INTERPRETER_LANGUAGES
        or not 1 <= len(callers) <= MAX_CALLER_LANGUAGES
        or len(set(callers)) != len(callers)
        or any(code not in INTERPRETER_LANGUAGES for code in callers)
        or working in callers
    ):
        raise invalid
    automatic = working in AUTO_DIRECTION_LANGUAGES and all(
        code in AUTO_DIRECTION_LANGUAGES for code in callers
    )
    if not automatic and len(callers) != 1:
        raise invalid
    result: dict[str, Any] = {"working_language": working, "caller_languages": callers}
    if translation.get("glossary"):
        try:
            result["glossary"] = validate_glossary(translation["glossary"])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"invalid_glossary: {exc}") from exc
    return result


class AgentIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    system_prompt: str = ""
    enabled_languages: list[str] = ["en", "ru"]
    stt_provider_id: UUID | None = None
    llm_provider_id: UUID | None = None
    tts_provider_id: UUID | None = None
    voice_id: str | None = None
    voice_profile_id: UUID | None = None
    mode: str = "agent"
    interpreter: InterpreterConfig | None = None
    translation: dict[str, Any] | None = None
    status: str = "enabled"

    @field_validator("enabled_languages", mode="after")
    @classmethod
    def validate_enabled_languages_field(cls, value: list[str]) -> list[str]:
        return validate_enabled_languages(value)

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value not in ("agent", "interpreter"):
            raise ValueError("mode must be 'agent' or 'interpreter'")
        return value

    @model_validator(mode="after")
    def validate_interpreter_config_required(self) -> "AgentIn":
        if self.mode == "interpreter" and self.interpreter is None:
            raise ValueError("interpreter_config_required")
        if self.mode == "agent" and self.interpreter is not None:
            raise ValueError("interpreter_config_not_allowed")
        return self


class Agent(AgentIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: Any
    updated_at: Any


class AgentPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    system_prompt: str | None = None
    enabled_languages: list[str] | None = None
    stt_provider_id: UUID | None = None
    llm_provider_id: UUID | None = None
    tts_provider_id: UUID | None = None
    voice_id: str | None = None
    voice_profile_id: UUID | None = None
    mode: str | None = None
    interpreter: InterpreterConfig | None = None
    translation: dict[str, Any] | None = None
    status: str | None = None

    @field_validator("name", "enabled_languages", "status", mode="before")
    @classmethod
    def reject_null_for_required_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value

    @field_validator("enabled_languages", mode="after")
    @classmethod
    def validate_enabled_languages_field(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else validate_enabled_languages(value)

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: str | None) -> str | None:
        if value is not None and value not in ("agent", "interpreter"):
            raise ValueError("mode must be 'agent' or 'interpreter'")
        return value


class ProviderIn(BaseModel):
    kind: str = Field(pattern="^(stt|llm|tts)$")
    provider: str = Field(min_length=1)
    configuration: dict[str, Any] = Field(default_factory=dict)
    secret_reference: str | None = None
    enabled: bool = True

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class Provider(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    kind: str
    provider: str
    configuration: dict[str, Any]
    secret_reference: str | None
    enabled: bool

def agent_response(agent: AgentRecord) -> Agent:
    return Agent.model_validate(agent)


def _load_in_org(db: Session, ctx: AuthContext | None, model: type, row_id: str):
    """Row by id within ``ctx``'s organisation. Another organisation's row comes back as
    ``None``, exactly like a missing one, so callers answer both with the same 422."""

    if ctx is None:
        return db.scalars(select(model).where(model.id == row_id)).first()
    return db.scalars(scoped(db, ctx, model).where(model.id == row_id)).first()


def check_voice_id_ownership(db: Session, ctx: AuthContext, voice_id: str | None) -> None:
    """409 ``voice_not_owned`` when ``voice_id`` is another organisation's cloned voice."""

    if not voice_id or ctx.org_id is None:
        return
    with bypass(db):
        owners = set(
            db.scalars(
                select(Voice.organization_id).where(
                    Voice.source == "clone", Voice.provider_voice_id == voice_id
                )
            )
        )
    if owners and ctx.org_id not in owners:
        raise HTTPException(status_code=409, detail={"code": "voice_not_owned"})


def provider_ids_for_agent(
    values: dict[str, Any],
    db: Session,
    engine_explicit: bool | None = None,
    ctx: AuthContext | None = None,
) -> dict[str, Any]:
    if ctx is not None:
        check_voice_id_ownership(db, ctx, values.get("voice_id"))
    for field, kind in (
        ("stt_provider_id", "stt"),
        ("llm_provider_id", "llm"),
        ("tts_provider_id", "tts"),
    ):
        if field not in values:
            continue
        provider_id = values[field]
        if provider_id is None:
            continue
        provider = _load_in_org(db, ctx, ProviderConfig, str(provider_id))
        if provider is None or provider.kind != kind or not provider.enabled:
            raise HTTPException(status_code=422, detail=f"invalid_{field}")
        values[field] = str(provider_id)
    voice_profile_id = values.get("voice_profile_id")
    if voice_profile_id is not None:
        voice = _load_in_org(db, ctx, Voice, str(voice_profile_id))
        if voice is None or voice.status in {"disabled", "deleting", "deleted"}:
            raise HTTPException(status_code=422, detail="invalid_voice_profile_id")
        values["voice_profile_id"] = str(voice_profile_id)

    # Convert interpreter config to dict if present
    interpreter = values.get("interpreter")
    validate_interpreter_engine(interpreter)
    if interpreter is not None and isinstance(interpreter, InterpreterConfig):
        values["interpreter"] = interpreter = interpreter.model_dump()
    if isinstance(interpreter, dict):
        if not interpreter.get("glossary"):
            interpreter.pop("glossary", None)
        if engine_explicit is None:
            # PATCH bodies are dumped with exclude_unset, so a default engine is absent.
            engine_explicit = "engine" in interpreter
        values["interpreter"] = resolve_interpreter_engine(interpreter, engine_explicit)

    return values

@router.post("/agents", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(
    payload: AgentIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> Agent:
    data = provider_ids_for_agent(
        payload.model_dump(),
        db,
        engine_explicit=payload.interpreter is not None
        and "engine" in payload.interpreter.model_fields_set,
        ctx=ctx,
    )
    data["translation"] = validate_agent_translation(
        data["mode"], data.get("translation")
    )
    record = AgentRecord(**data, organization_id=ctx.org_id)
    db.add(record)
    db.commit()
    db.refresh(record)
    return agent_response(record)


@router.get("/agents", response_model=list[Agent])
async def list_agents(
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> list[Agent]:
    return [
        agent_response(agent)
        for agent in db.scalars(
            scoped(db, ctx, AgentRecord).where(AgentRecord.system.is_(False))
        ).all()
    ]


@router.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> Agent:
    record = get_public_agent(db, ctx, agent_id)
    return agent_response(record)


@router.patch("/agents/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: UUID,
    payload: AgentPatch,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> Agent:
    record = get_public_agent(db, ctx, agent_id)

    changes = provider_ids_for_agent(
        payload.model_dump(exclude_unset=True), db, ctx=ctx
    )

    # Validate merged state for mode/interpreter
    current_mode = changes.get("mode", record.mode)
    current_interpreter = changes.get("interpreter", record.interpreter)

    if current_mode == "interpreter" and current_interpreter is None:
        raise HTTPException(status_code=422, detail="interpreter_config_required")
    if current_mode == "agent" and current_interpreter is not None:
        raise HTTPException(status_code=422, detail="interpreter_config_not_allowed")
    if "translation" in changes or "mode" in changes:
        changes["translation"] = validate_agent_translation(
            current_mode, changes.get("translation", record.translation)
        )

    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return agent_response(record)


@router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> None:
    record = get_public_agent(db, ctx, agent_id)
    record.status = "disabled"
    db.commit()


@router.post("/providers", response_model=Provider, status_code=status.HTTP_201_CREATED)
async def create_provider(
    payload: ProviderIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> Provider:
    record = ProviderConfig(**payload.model_dump(), organization_id=ctx.org_id)
    db.add(record)
    db.commit()
    db.refresh(record)
    return Provider.model_validate(record)


@router.get("/providers", response_model=list[Provider])
async def list_providers(
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> list[Provider]:
    return [
        Provider.model_validate(item)
        for item in db.scalars(scoped(db, ctx, ProviderConfig)).all()
    ]
