# Backend Voice Labs workspace

This local repository groups the independently structured Platform, Runtime, and
Evaluation projects. Their project-specific `AGENTS.md`, `pyproject.toml`, lockfiles,
tests, migrations, and conventions remain authoritative within each member directory.

Do not move, delete, or edit the original sibling repositories while working here.
Do not add remotes, commit, push, deploy, or create submodules without explicit
authorization. Keep provider credentials and private recordings out of the repository.

Run each member's own checks from its directory:

- `cd platform && python -m pytest -q`
- `cd runtime && python -m pytest -q`
- `cd evals && python -m pytest -q`
- Platform migrations: `cd platform && python -m alembic upgrade head` against a
  configured disposable/local database.

The root uv workspace is for unified dependency resolution; member lockfiles are
preserved as source artifacts. Use the root `uv.lock` for workspace resolution when
available, and do not replace member project configuration merely to satisfy the
workspace.
