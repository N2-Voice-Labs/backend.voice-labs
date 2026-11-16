from src.platform.validation import contains_secret_key


def test_token_counts_are_settings_not_secrets():
    assert not contains_secret_key({"model": "gemini-3.5-flash-lite", "max_tokens": 160})
    assert not contains_secret_key({"nested": {"max_output_tokens": 300}})


def test_real_secret_names_are_still_rejected():
    assert contains_secret_key({"api_key": "x"})
    assert contains_secret_key({"access_token": "x"})
    assert contains_secret_key({"settings": {"client_secret": "x"}})
