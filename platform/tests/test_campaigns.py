import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app


def phone() -> str:
    return f"+1{uuid4().int % 10**10:010d}"


def make_agent(client: TestClient) -> dict:
    return client.post("/agents", json={"name": f"agent-{uuid4()}"}).json()


def make_deployment(client: TestClient, agent_id: str, channel: str = "sip") -> dict:
    body = {"agent_id": agent_id, "name": f"dep-{uuid4()}", "channel": channel}
    if channel == "sip":
        body["phone_number"] = phone()
    response = client.post("/deployments", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def make_campaign(client: TestClient, **overrides) -> dict:
    """Create a draft campaign; unless overridden, with its own agent and SIP deployment."""
    if "agent_id" not in overrides:
        overrides["agent_id"] = make_agent(client)["id"]
    if "deployment_id" not in overrides:
        overrides["deployment_id"] = make_deployment(client, overrides["agent_id"])["id"]
    response = client.post("/campaigns", json={"name": f"camp-{uuid4()}", **overrides})
    assert response.status_code == 201, response.text
    return response.json()


def add_targets(client: TestClient, campaign_id: str, count: int = 1) -> list[str]:
    numbers = [phone() for _ in range(count)]
    response = client.post(
        f"/campaigns/{campaign_id}/targets",
        json={"targets": [{"phone_number": number} for number in numbers]},
    )
    assert response.status_code == 200, response.text
    return numbers


def set_status(client: TestClient, campaign_id: str, status: str):
    return client.post(f"/campaigns/{campaign_id}/status", json={"status": status})


def test_campaign_create_get_list_patch() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        campaign = make_campaign(
            client,
            agent_id=agent["id"],
            schedule_start="2030-01-01T09:00:00Z",
            schedule_end="2030-01-01T17:00:00Z",
            timezone="Europe/Berlin",
            max_attempts=3,
            configuration={"retry_delay_minutes": 30},
        )
        assert campaign["status"] == "draft"
        assert campaign["timezone"] == "Europe/Berlin"
        assert campaign["max_attempts"] == 3
        assert campaign["target_counts"] == {}

        defaults = make_campaign(client)
        assert defaults["timezone"] == "UTC"
        assert defaults["max_attempts"] == 1

        add_targets(client, campaign["id"], 2)
        detail = client.get(f"/campaigns/{campaign['id']}").json()
        assert detail["target_counts"] == {"pending": 2}

        listed = client.get("/campaigns", params={"agent_id": agent["id"]}).json()
        assert [item["id"] for item in listed] == [campaign["id"]]
        assert client.get(
            "/campaigns", params={"agent_id": agent["id"], "status": "running"}
        ).json() == []
        assert len(client.get("/campaigns", params={"status": "draft"}).json()) >= 2

        patched = client.patch(
            f"/campaigns/{campaign['id']}",
            json={"name": "renamed", "max_attempts": 5, "timezone": "UTC"},
        )
        assert patched.status_code == 200
        assert patched.json()["name"] == "renamed"
        assert patched.json()["max_attempts"] == 5
        assert patched.json()["target_counts"] == {"pending": 2}


def test_campaign_not_found() -> None:
    with TestClient(app) as client:
        missing = str(uuid4())
        for response in (
            client.get(f"/campaigns/{missing}"),
            client.patch(f"/campaigns/{missing}", json={"name": "x"}),
            client.get(f"/campaigns/{missing}/targets"),
            client.post(
                f"/campaigns/{missing}/targets",
                json={"targets": [{"phone_number": phone()}]},
            ),
            set_status(client, missing, "scheduled"),
        ):
            assert response.status_code == 404
            assert response.json()["detail"] == "campaign_not_found"


def test_create_validation_errors() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        other_agent = make_agent(client)
        sip = make_deployment(client, agent["id"])
        web = make_deployment(client, agent["id"], channel="web")
        base = {"name": "c", "agent_id": agent["id"], "deployment_id": sip["id"]}

        def post(**overrides):
            return client.post("/campaigns", json={**base, **overrides})

        response = post(timezone="Mars/Olympus")
        assert (response.status_code, response.json()["detail"]) == (422, "invalid_timezone")
        response = post(deployment_id=web["id"])
        assert response.status_code == 422
        assert response.json()["detail"] == "campaign_requires_sip_deployment"
        response = post(agent_id=other_agent["id"])
        assert (response.status_code, response.json()["detail"]) == (
            409,
            "deployment_agent_mismatch",
        )
        assert post(agent_id=str(uuid4())).json()["detail"] == "invalid_agent_id"
        assert post(deployment_id=str(uuid4())).json()["detail"] == "invalid_deployment_id"
        response = post(
            schedule_start="2030-01-02T00:00:00Z", schedule_end="2030-01-01T00:00:00Z"
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_schedule"
        assert post(max_attempts=0).status_code == 422
        assert post(max_attempts=11).status_code == 422
        assert post(name="").status_code == 422
        assert post(configuration={"api_key": "x"}).status_code == 422


def test_patch_validation_and_editability() -> None:
    with TestClient(app) as client:
        campaign = make_campaign(client, schedule_start="2030-01-01T09:00:00Z")
        url = f"/campaigns/{campaign['id']}"

        assert client.patch(url, json={"timezone": "Nope/Nope"}).json()["detail"] == "invalid_timezone"
        response = client.patch(url, json={"schedule_end": "2029-12-31T00:00:00Z"})
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_schedule"
        assert client.patch(url, json={"max_attempts": 99}).status_code == 422
        assert client.patch(url, json={"name": None}).status_code == 422
        assert client.patch(url, json={"configuration": {"token": "x"}}).status_code == 422

        add_targets(client, campaign["id"])
        assert set_status(client, campaign["id"], "scheduled").status_code == 200
        assert client.patch(url, json={"name": "while-scheduled"}).status_code == 200
        assert set_status(client, campaign["id"], "running").status_code == 200

        response = client.patch(url, json={"name": "too-late"})
        assert response.status_code == 409
        assert response.json()["detail"] == "campaign_not_editable"


def test_add_targets_deduplicates() -> None:
    with TestClient(app) as client:
        campaign = make_campaign(client)
        url = f"/campaigns/{campaign['id']}/targets"
        a, b, c = phone(), phone(), phone()

        response = client.post(
            url,
            json={
                "targets": [
                    {"phone_number": a, "name": "Ann", "variables": {"plan": "pro"}},
                    {"phone_number": b},
                    {"phone_number": a, "name": "Ann again"},
                ]
            },
        )
        assert response.status_code == 200
        assert response.json() == {"added": 2, "skipped_duplicates": 1}

        response = client.post(
            url, json={"targets": [{"phone_number": b}, {"phone_number": c}]}
        )
        assert response.json() == {"added": 1, "skipped_duplicates": 1}

        targets = client.get(url).json()
        assert sorted(item["phone_number"] for item in targets) == sorted([a, b, c])
        ann = next(item for item in targets if item["phone_number"] == a)
        assert (ann["name"], ann["variables"]) == ("Ann", {"plan": "pro"})
        assert (ann["status"], ann["attempts"]) == ("pending", 0)

        # Same number is allowed in a different campaign.
        other = make_campaign(client)
        response = client.post(
            f"/campaigns/{other['id']}/targets", json={"targets": [{"phone_number": a}]}
        )
        assert response.json() == {"added": 1, "skipped_duplicates": 0}


def test_add_targets_validation_and_state() -> None:
    with TestClient(app) as client:
        campaign = make_campaign(client)
        url = f"/campaigns/{campaign['id']}/targets"

        assert client.post(url, json={"targets": []}).status_code == 422
        assert client.post(url, json={"targets": [{"phone_number": "12345"}]}).status_code == 422
        too_many = [{"phone_number": f"+1{n:010d}"} for n in range(1, 1002)]
        assert client.post(url, json={"targets": too_many}).status_code == 422
        exactly_max = [{"phone_number": f"+2{n:010d}"} for n in range(1000)]
        response = client.post(url, json={"targets": exactly_max})
        assert response.json() == {"added": 1000, "skipped_duplicates": 0}

        # Allowed while scheduled and paused, rejected once running/cancelled.
        assert set_status(client, campaign["id"], "scheduled").status_code == 200
        add_targets(client, campaign["id"])
        set_status(client, campaign["id"], "running")
        response = client.post(url, json={"targets": [{"phone_number": phone()}]})
        assert response.status_code == 409
        assert response.json()["detail"] == "campaign_not_editable"
        set_status(client, campaign["id"], "paused")
        add_targets(client, campaign["id"])
        set_status(client, campaign["id"], "cancelled")
        response = client.post(url, json={"targets": [{"phone_number": phone()}]})
        assert response.status_code == 409


def test_list_targets_pagination_and_status_filter() -> None:
    with TestClient(app) as client:
        campaign = make_campaign(client)
        url = f"/campaigns/{campaign['id']}/targets"
        numbers = add_targets(client, campaign["id"], 5)

        first = client.get(url, params={"limit": 2}).json()
        second = client.get(url, params={"limit": 2, "offset": 2}).json()
        rest = client.get(url, params={"limit": 2, "offset": 4}).json()
        paged = [item["phone_number"] for item in first + second + rest]
        assert sorted(paged) == sorted(numbers)
        assert len(set(paged)) == 5
        assert (len(first), len(second), len(rest)) == (2, 2, 1)

        assert len(client.get(url, params={"status": "pending"}).json()) == 5
        assert client.get(url, params={"status": "completed"}).json() == []
        assert client.get(url, params={"limit": 501}).status_code == 422
        assert client.get(url, params={"limit": 0}).status_code == 422
        assert client.get(url, params={"offset": -1}).status_code == 422
        assert client.get(url, params={"limit": 500}).status_code == 200


def test_status_transitions_happy_path() -> None:
    with TestClient(app) as client:
        campaign = make_campaign(client)
        add_targets(client, campaign["id"])
        for current in (
            "scheduled",
            "draft",
            "scheduled",
            "running",
            "paused",
            "running",
            "completed",
        ):
            response = set_status(client, campaign["id"], current)
            assert response.status_code == 200, (current, response.text)
            assert response.json()["status"] == current
            assert response.json()["target_counts"] == {"pending": 1}


def test_cancel_from_every_non_terminal_state() -> None:
    with TestClient(app) as client:
        for path in (
            [],
            ["scheduled"],
            ["scheduled", "running"],
            ["scheduled", "running", "paused"],
        ):
            campaign = make_campaign(client)
            add_targets(client, campaign["id"])
            for step in path:
                assert set_status(client, campaign["id"], step).status_code == 200
            response = set_status(client, campaign["id"], "cancelled")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"


def test_invalid_status_transitions() -> None:
    with TestClient(app) as client:
        campaign = make_campaign(client)
        add_targets(client, campaign["id"])
        cid = campaign["id"]

        def rejected(target: str) -> None:
            response = set_status(client, cid, target)
            assert response.status_code == 409, (target, response.text)
            assert response.json()["detail"] == "invalid_campaign_status_transition"

        rejected("running")  # draft -> running
        rejected("paused")
        rejected("completed")
        rejected("draft")  # no-op
        assert set_status(client, cid, "scheduled").status_code == 200
        rejected("paused")
        rejected("completed")
        set_status(client, cid, "running")
        rejected("draft")
        rejected("scheduled")
        set_status(client, cid, "completed")
        for target in ("draft", "scheduled", "running", "paused", "cancelled"):
            rejected(target)

        cancelled = make_campaign(client)
        set_status(client, cancelled["id"], "cancelled")
        cid = cancelled["id"]
        rejected("draft")
        rejected("scheduled")

        assert set_status(client, cid, "bogus").status_code == 422


def test_scheduling_requires_targets_and_active_deployment() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        deployment = make_deployment(client, agent["id"])
        campaign = make_campaign(
            client, agent_id=agent["id"], deployment_id=deployment["id"]
        )

        response = set_status(client, campaign["id"], "scheduled")
        assert response.status_code == 409
        assert response.json()["detail"] == "campaign_has_no_targets"

        add_targets(client, campaign["id"])
        client.patch(f"/deployments/{deployment['id']}", json={"status": "paused"})
        response = set_status(client, campaign["id"], "scheduled")
        assert response.status_code == 409
        assert response.json()["detail"] == "deployment_not_active"

        client.patch(f"/deployments/{deployment['id']}", json={"status": "active"})
        assert set_status(client, campaign["id"], "scheduled").status_code == 200

        # Resuming also needs an active deployment; cancelling does not.
        client.patch(f"/deployments/{deployment['id']}", json={"status": "paused"})
        response = set_status(client, campaign["id"], "running")
        assert response.status_code == 409
        assert response.json()["detail"] == "deployment_not_active"
        assert set_status(client, campaign["id"], "cancelled").status_code == 200
