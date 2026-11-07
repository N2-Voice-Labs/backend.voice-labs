"""The cinematic-dubbing option: off by default and invisible when off, validated when on."""

import os

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.platform.agents import InterpreterConfig
from src.platform.translations import TranslationIn, interpreter_config


def payload(**extra) -> TranslationIn:
    return TranslationIn(languages=["ko", "en"], creator={"language": "ko"}, **extra)


def test_dubbing_is_absent_from_the_config_unless_set() -> None:
    off = interpreter_config(payload())
    assert "dubbing" not in off  # so the hidden agent's hash is what it was before the option
    assert interpreter_config(payload(dubbing=False)) == off
    assert interpreter_config(payload(dubbing=None)) == off


def test_dubbing_is_passed_to_the_runtime_when_set() -> None:
    assert interpreter_config(payload(dubbing=True))["dubbing"] is True
    assert interpreter_config(payload(dubbing="subtle"))["dubbing"] == "subtle"
    assert interpreter_config(payload(dubbing="cinematic"))["dubbing"] == "cinematic"


def test_unknown_profiles_are_rejected() -> None:
    with pytest.raises(ValidationError):
        payload(dubbing="deafening")


def test_dubbing_and_hear_original_exclude_each_other() -> None:
    with pytest.raises(HTTPException) as error:
        interpreter_config(payload(dubbing=True, hear_original=True))
    assert error.value.status_code == 422 and error.value.detail == "invalid_interpreter_config"
    with pytest.raises(ValidationError):
        InterpreterConfig(languages=["en", "ru"], dubbing="balanced", hear_original=True)
    assert InterpreterConfig(languages=["en", "ru"], dubbing=False, hear_original=True).hear_original
