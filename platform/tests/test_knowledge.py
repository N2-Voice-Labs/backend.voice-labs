import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app


def make_doc(client: TestClient, **overrides) -> dict:
    body = {
        "title": f"Doc {uuid4().hex[:8]}",
        "content": "Opening hours are 9 to 5.",
        "source": "https://example.test/faq",
        "metadata": {"lang": "en"},
        **overrides,
    }
    response = client.post("/knowledge/documents", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_document_create_get_and_list_omits_content() -> None:
    with TestClient(app) as client:
        doc = make_doc(client)
        assert doc["version"] == 1 and doc["enabled"] is True
        assert doc["metadata"] == {"lang": "en"}
        assert doc["content_length"] == len("Opening hours are 9 to 5.")
        assert client.get(f"/knowledge/documents/{doc['id']}").json() == doc

        listed = client.get("/knowledge/documents").json()
        row = next(d for d in listed if d["id"] == doc["id"])
        assert "content" not in row and row["content_length"] == doc["content_length"]

        default = client.post(
            "/knowledge/documents", json={"title": "T", "content": "c"}
        ).json()
        assert default["source"] == "" and default["metadata"] == {}


def test_document_validation_errors() -> None:
    with TestClient(app) as client:
        base = {"title": "T", "content": "c"}
        for bad in (
            {"title": ""},
            {"title": "x" * 301},
            {"source": "x" * 1025},
            {"content": ""},
            {"content": "x" * 1_000_001},
            {"metadata": {"api_key": "k"}},
        ):
            assert (
                client.post("/knowledge/documents", json={**base, **bad}).status_code
                == 422
            ), list(bad)


def test_document_patch_versions_only_on_content_change() -> None:
    with TestClient(app) as client:
        doc = make_doc(client)
        url = f"/knowledge/documents/{doc['id']}"
        meta = client.patch(
            url, json={"title": "New", "source": "s", "metadata": {"a": 1}}
        ).json()
        assert meta["version"] == 1 and meta["title"] == "New"
        assert meta["source"] == "s" and meta["metadata"] == {"a": 1}
        same = client.patch(url, json={"content": doc["content"]}).json()
        assert same["version"] == 1
        changed = client.patch(url, json={"content": "Brand new text"}).json()
        assert changed["version"] == 2 and changed["content"] == "Brand new text"
        assert changed["content_length"] == len("Brand new text")
        for bad in ({"title": None}, {"content": ""}, {"metadata": {"token": "x"}}):
            assert client.patch(url, json=bad).status_code == 422


def test_document_delete_soft_and_filter() -> None:
    with TestClient(app) as client:
        doc = make_doc(client)
        assert client.delete(f"/knowledge/documents/{doc['id']}").status_code == 204
        assert client.get(f"/knowledge/documents/{doc['id']}").json()["enabled"] is False
        on = {d["id"] for d in client.get("/knowledge/documents?enabled=true").json()}
        off = {d["id"] for d in client.get("/knowledge/documents?enabled=false").json()}
        assert doc["id"] not in on and doc["id"] in off
        revived = client.patch(f"/knowledge/documents/{doc['id']}", json={"enabled": True})
        assert revived.json()["enabled"] is True


def test_document_not_found() -> None:
    with TestClient(app) as client:
        url = f"/knowledge/documents/{uuid4()}"
        for response in (
            client.get(url),
            client.patch(url, json={"title": "x"}),
            client.delete(url),
        ):
            assert response.status_code == 404
            assert response.json()["detail"] == "knowledge_document_not_found"


def test_agent_knowledge_replace_dedupe_and_errors() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Knowledge agent"}).json()
        url = f"/agents/{agent['id']}/knowledge"
        assert client.get(url).json() == []
        a, b, off = make_doc(client), make_doc(client), make_doc(client)
        client.delete(f"/knowledge/documents/{off['id']}")

        put = client.put(url, json={"document_ids": [a["id"], b["id"], a["id"]]})
        assert put.status_code == 200 and len(put.json()) == 2
        assert all("content" not in d for d in put.json())
        assert client.put(url, json={"document_ids": [b["id"]]}).status_code == 200
        assert [d["id"] for d in client.get(url).json()] == [b["id"]]

        for ids in ([str(uuid4())], [off["id"]], [b["id"], str(uuid4())]):
            bad = client.put(url, json={"document_ids": ids})
            assert bad.status_code == 422
            assert bad.json()["detail"] == "invalid_knowledge_document_id"
        assert [d["id"] for d in client.get(url).json()] == [b["id"]]
        assert client.put(url, json={"document_ids": []}).json() == []

        missing = f"/agents/{uuid4()}/knowledge"
        for response in (
            client.get(missing),
            client.put(missing, json={"document_ids": []}),
        ):
            assert response.status_code == 404
            assert response.json()["detail"] == "agent_not_found"
