# Platform agent instructions

This repository is the Voice Labs control plane, based on the selected FastAPI ORM
template. It owns durable agents, provider references, conversations, messages, voice
sessions, and runtime event ingestion. It must not process realtime audio.

Use SQLAlchemy/Alembic/PostgreSQL conventions. Keep provider secrets out of API payloads.
Run `python -m pytest -q` and `python -m alembic upgrade head` against a configured local
database before handing off. Do not add Git remotes, commit, push, or deploy.
