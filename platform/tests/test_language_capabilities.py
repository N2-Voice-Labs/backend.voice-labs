"""Language capability table (ST-20) and glossary engine routing (ST-41)."""

import os

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient

from src.platform.main import app, resolve_interpreter_engine
from src.platform.validation import (
    AUTO_DIRECTION_LANGUAGES,
    INTERPRETER_LANGUAGES,
    LANGUAGE_CAPABILITIES,
    REALTIME_TRANSLATE_OUTPUT_LANGUAGES,
)

REALTIME_MESSAGE = "Glossary is not supported by the realtime engine; use step-by-step (cascaded)"
GLOSSARY = [{"source": "residence permit", "translations": {"ru": "вид на жительство"}}]


def test_table_invariants() -> None:
    assert AUTO_DIRECTION_LANGUAGES <= INTERPRETER_LANGUAGES
    for code in AUTO_DIRECTION_LANGUAGES:
        assert LANGUAGE_CAPABILITIES[code]["stt_fixed"], code
    assert "ko" not in AUTO_DIRECTION_LANGUAGES
    assert "ko" in INTERPRETER_LANGUAGES
    assert AUTO_DIRECTION_LANGUAGES == {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}
    assert {"id", "vi"} <= REALTIME_TRANSLATE_OUTPUT_LANGUAGES
    assert not {"id", "vi"} & INTERPRETER_LANGUAGES


def test_ko_with_auto_direction_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "KO auto",
                "mode": "interpreter",
                "interpreter": {"languages": ["ko", "en"], "direction": "auto"},
            },
        )
    assert response.status_code == 422
    assert "code-switching" in str(response.json()["detail"])


def test_unsupported_agent_language_is_rejected_with_its_name() -> None:
    with TestClient(app) as client:
        for bad in ("xx", "id"):
            response = client.post(
                "/agents", json={"name": "A", "enabled_languages": ["en", bad]}
            )
            assert response.status_code == 422, bad
            assert f"unsupported language in enabled_languages: {bad}" in str(
                response.json()["detail"]
            )
        ok = client.post("/agents", json={"name": "A", "enabled_languages": ["en", "KO", "tr"]})
        assert ok.status_code == 201
        assert ok.json()["enabled_languages"] == ["en", "ko", "tr"]
        patched = client.patch(f"/agents/{ok.json()['id']}", json={"enabled_languages": ["zz"]})
        assert patched.status_code == 422
        assert "zz" in str(patched.json()["detail"])
        assert client.patch(
            f"/agents/{ok.json()['id']}", json={"enabled_languages": ["pl", "en"]}
        ).status_code == 200


def _interpreter(**extra) -> dict:
    return {"languages": ["en", "ru"], "direction": "one_way", "latency_mode": "instant", **extra}


def test_resolve_instant_with_glossary_stays_cascaded() -> None:
    config = _interpreter(glossary=GLOSSARY)
    assert resolve_interpreter_engine(dict(config), False).get("engine") != "realtime"
    assert resolve_interpreter_engine(_interpreter(), False)["engine"] == "realtime"


def test_instant_agent_with_glossary_is_stored_cascaded() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={"name": "G", "mode": "interpreter", "interpreter": _interpreter(glossary=GLOSSARY)},
        )
        assert response.status_code == 201
        assert response.json()["interpreter"]["engine"] == "cascaded"
        plain = client.post(
            "/agents", json={"name": "P", "mode": "interpreter", "interpreter": _interpreter()}
        )
        assert plain.json()["interpreter"]["engine"] == "realtime"
        # Adding a glossary by PATCH (the interpreter block is replaced) goes cascaded.
        patched = client.patch(
            f"/agents/{plain.json()['id']}", json={"interpreter": _interpreter(glossary=GLOSSARY)}
        )
        assert patched.status_code == 200
        assert patched.json()["interpreter"]["engine"] == "cascaded"


def test_explicit_realtime_with_glossary_is_rejected() -> None:
    with TestClient(app) as client:
        body = {
            "name": "R",
            "mode": "interpreter",
            "interpreter": _interpreter(engine="realtime", glossary=GLOSSARY),
        }
        response = client.post("/agents", json=body)
        assert response.status_code == 422
        assert response.json()["detail"] == REALTIME_MESSAGE
        agent = client.post(
            "/agents",
            json={**body, "interpreter": _interpreter(engine="realtime")},
        ).json()
        patched = client.patch(
            f"/agents/{agent['id']}",
            json={"interpreter": _interpreter(engine="realtime", glossary=GLOSSARY)},
        )
        assert patched.status_code == 422
        assert patched.json()["detail"] == REALTIME_MESSAGE
