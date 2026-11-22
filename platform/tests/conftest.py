import os

# Test modules share one process and database and leak sessions, so the background
# reaper and the session limits are off unless a test sets them explicitly.
for _name in (
    "PLATFORM_REAPER_INTERVAL_SECS",
    "PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG",
    "PLATFORM_SESSION_CREATE_RATE",
    "PLATFORM_DIAL_PER_ORG_HOUR",
):
    os.environ.setdefault(_name, "0")

# Most tests run without PLATFORM_API_KEYS and rely on the ``dev`` owner principal, which
# now exists only when auth is explicitly disabled. Tests of the fail-closed behaviour
# remove this variable themselves.
os.environ.setdefault("PLATFORM_AUTH_DISABLED", "1")
