#!/usr/bin/env python3
"""Create or update a demo agent through the Platform HTTP API."""

import argparse
import json
from pathlib import Path
import re
from typing import Any

import httpx


def read_agent_files(agent_dir: Path, llm_model: str | None = None) -> dict[str, Any]:
    """Parse the local agent definition, prompt and eligible Markdown knowledge files."""
    config = json.loads((agent_dir / "agent.json").read_text(encoding="utf-8"))
    config.setdefault("llm", {}).setdefault("settings", {})
    if llm_model:
        config["llm"]["settings"]["model"] = llm_model
    knowledge = []
    for path in sorted((agent_dir / "knowledge").glob("*.md")):
        if path.name.startswith("_") or path.name.upper() == "README.MD":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            source = re.search(r"(?m)^Source:\s*(\S+)\s*$", text)
            knowledge.append({"title": path.stem.replace("_", " "), "content": text, "source": source.group(1) if source else ""})
    return {"config": config, "prompt": (agent_dir / "prompt.md").read_text(encoding="utf-8").strip(), "knowledge": knowledge}


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    if response.status_code == 204:
        return None
    return response.json()


def seed(base_url: str, api_key: str | None, data: dict[str, Any], dry_run: bool = False) -> str:
    config = data["config"]
    if dry_run:
        print(json.dumps({"agent": config["name"], "providers": config, "knowledge": data["knowledge"]}, indent=2))
        return "<dry-run>"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=30) as client:
        providers: dict[str, str] = {}
        known = _request(client, "GET", "/providers")
        for kind in ("stt", "llm", "tts"):
            spec = config[kind]
            settings = spec.get("settings", {}).copy()
            existing = next((p for p in known if p["kind"] == kind and p["provider"] == spec["provider"] and p["configuration"] == settings and p["enabled"]), None)
            if existing is None:
                existing = _request(client, "POST", "/providers", json={"kind": kind, "provider": spec["provider"], "configuration": settings, "enabled": True})
                known.append(existing)
            providers[kind] = existing["id"]

        agents = _request(client, "GET", "/agents")
        body = {"name": config["name"], "description": config.get("description", ""), "system_prompt": data["prompt"],
                "enabled_languages": config["enabled_languages"], "stt_provider_id": providers["stt"],
                "llm_provider_id": providers["llm"], "tts_provider_id": providers["tts"], "voice_id": config.get("voice_id"), "status": "enabled"}
        agent = next((item for item in agents if item["name"] == config["name"]), None)
        agent = _request(client, "PATCH", f"/agents/{agent['id']}", json=body) if agent else _request(client, "POST", "/agents", json=body)
        agent_id = agent["id"]

        docs = _request(client, "GET", "/knowledge/documents")
        linked_ids = []
        for document in data["knowledge"]:
            existing = next((item for item in docs if item["title"] == document["title"]), None)
            payload = {**document, "metadata": {}}
            if existing:
                updated = _request(client, "PATCH", f"/knowledge/documents/{existing['id']}", json=payload)
                linked_ids.append(updated["id"])
            else:
                created = _request(client, "POST", "/knowledge/documents", json=payload)
                docs.append(created)
                linked_ids.append(created["id"])
        _request(client, "PUT", f"/agents/{agent_id}/knowledge", json={"document_ids": linked_ids})

        tools = _request(client, "GET", "/tools")
        tool_ids = []
        for name in config.get("tools", []):
            tool = next((item for item in tools if item["name"] == name), None)
            if tool is None and name == "transfer_to_human":
                tool = _request(client, "POST", "/tools", json={"name": name, "description": "Connect the caller to a human admissions officer.", "tool_type": "builtin", "input_schema": {"type": "object", "properties": {}}, "configuration": {}})
            if tool:
                tool_ids.append(tool["id"])
        _request(client, "PUT", f"/agents/{agent_id}/tools", json={"tool_ids": tool_ids})
    return agent_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-url", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--agent-dir", type=Path, default=Path("agents/inha-admissions"))
    parser.add_argument("--llm-model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = read_agent_files(args.agent_dir, args.llm_model)
    agent_id = seed(args.platform_url, args.api_key, data, args.dry_run)
    print(f"Agent ID: {agent_id}")
    print(f"PLATFORM_DEMO_AGENTS={data['config'].get('demo_key', data['config']['name'])}={agent_id}")


if __name__ == "__main__":
    main()
