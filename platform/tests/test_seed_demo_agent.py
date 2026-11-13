import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seed_demo_agent.py"
spec = importlib.util.spec_from_file_location("seed_demo_agent", SCRIPT)
seed_demo_agent = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(seed_demo_agent)


def test_read_agent_files_skips_templates_and_applies_model_override(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "agent.json").write_text(
        '{"name":"Demo","llm":{"settings":{"model":"old"}}}', encoding="utf-8"
    )
    (tmp_path / "prompt.md").write_text("Prompt", encoding="utf-8")
    (tmp_path / "knowledge" / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "knowledge" / "_template.md").write_text("template", encoding="utf-8")
    (tmp_path / "knowledge" / "admissions.md").write_text(
        "Source: https://example.edu/admissions\nRetrieved: 2026-11-12\n\nContent", encoding="utf-8"
    )

    parsed = seed_demo_agent.read_agent_files(tmp_path, "gemini-custom")
    assert parsed["prompt"] == "Prompt"
    assert parsed["config"]["llm"]["settings"]["model"] == "gemini-custom"
    assert parsed["knowledge"] == [{
        "title": "admissions", "source": "https://example.edu/admissions",
        "content": "Source: https://example.edu/admissions\nRetrieved: 2026-11-12\n\nContent",
    }]
