import os

# Test modules share one process and database and leak sessions, so the background
# reaper and the session limits are off unless a test sets them explicitly.
for _name in (
    "PLATFORM_REAPER_INTERVAL_SECS",
    "PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG",
    "PLATFORM_SESSION_CREATE_RATE",
):
    os.environ.setdefault(_name, "0")
