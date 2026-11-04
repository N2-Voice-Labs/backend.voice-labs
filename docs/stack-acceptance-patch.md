# Suggested fix for `stack.voice-labs/tests/test_voice_platform_acceptance.py`

Status: **APPLIED 2026-11-05, uncommitted** in the working tree of the sibling repository
`stack.voice-labs` (`tests/test_voice_platform_acceptance.py`), with the user's authorization. The
diff below is kept for reference. Result after applying: `6 passed, 3 skipped, 0 failed` (it was
`1 failed, 5 passed, 3 skipped`; the two skipped LiveKit tests skip because no LiveKit server could
be started, Docker's socket being permission-denied for this user, and the real-provider test is
opt-in). Earlier, applying it to a scratch copy gave the same result (lock-pinned environment). The same failure occurs on the unmodified
`HEAD`, so the test file is stale, not the backend.

## Why it fails

The test replaces the `httpx` name inside `src.platform.main` with a stub so Platform's calls to the
Runtime go to an in-process ASGI app. The session routes now live in `src/platform/sessions.py`, which
imports `httpx` itself, so the stub never applies there. `ensure-running` then resolves the real host
`runtime.test` and fails with `runtime_unavailable` (503).

## Patch (three call sites: the three `monkeypatch.setattr(platform_main, "httpx", ...)` blocks)

```diff
 from src.platform import main as platform_main  # noqa: E402
+from src.platform import sessions as platform_sessions  # noqa: E402
 ...
     monkeypatch.setattr(
-        platform_main,
+        platform_sessions,
         "httpx",
         SimpleNamespace(AsyncClient=RuntimeHttpClient ...),
     )
```

Apply the same one-word change (`platform_main` to `platform_sessions`) at each of the three places that
stub `"httpx"`. Check that any other module the test exercises that makes Runtime calls (for example
`deployments.py`, `sip.py`) is also covered, and stub it the same way if so.
