"""``simul isaac claims``: the operator's view of instance claims, and the way to clear them.

An agent locked out by a claim nothing would release (its server crashed, its
label was lost) had no recourse but to wait out the TTL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simul.cli.main import app
from simul.mcp import session_manager
from simul.mcp.session_manager import InstanceSession

runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(session_manager, "_DEFAULT_SESSION_DIR", tmp_path)
    InstanceSession(8226, tmp_path).register("robot-1", "long job", owner="o1")
    InstanceSession(8226, tmp_path).register("robot-2", "lighting", owner="o2")
    InstanceSession(8230, tmp_path).register("robot-3", "other box", owner="o3")
    return tmp_path


def _json(*args: str) -> tuple[int, dict]:
    result = runner.invoke(app, ["--json", "isaac", "claims", *args])
    return result.exit_code, json.loads(result.stdout)


def test_claims_lists_every_live_claim(store: Path) -> None:
    code, payload = _json()

    assert code == 0
    assert [(c["port"], c["agent_id"]) for c in payload["claims"]] == [
        (8226, "robot-1"),
        (8226, "robot-2"),
        (8230, "robot-3"),
    ]
    assert all(0 < c["expires_in_seconds"] <= 120 for c in payload["claims"])
    assert payload["claims"][0]["orphaned"] is False
    assert payload["session_dir"] == str(store)


def test_claims_filters_by_port_and_agent(store: Path) -> None:
    assert [c["agent_id"] for c in _json("--port", "8230")[1]["claims"]] == ["robot-3"]
    assert [c["agent_id"] for c in _json("--agent", "robot-2")[1]["claims"]] == ["robot-2"]


def test_claims_clear_removes_the_selected_claims(store: Path) -> None:
    code, payload = _json("--clear", "--port", "8226", "--agent", "robot-1")

    assert code == 0
    assert [(c["port"], c["agent_id"]) for c in payload["cleared"]] == [(8226, "robot-1")]
    assert [c["agent_id"] for c in _json()[1]["claims"]] == ["robot-2", "robot-3"]

    code, payload = _json("--clear")
    assert sorted(c["agent_id"] for c in payload["cleared"]) == ["robot-2", "robot-3"]
    assert _json()[1]["claims"] == []


def test_claims_reports_a_repaired_store(store: Path) -> None:
    (store / "8230.json").write_text("garbage")

    code, payload = _json()

    assert code == 0
    assert [c["port"] for c in payload["claims"]] == [8226, 8226]
    assert payload["problems"] == [{"port": 8230, "repaired_at": payload["problems"][0]["repaired_at"]}]


def test_claims_exits_non_zero_when_a_store_cannot_be_read(store: Path) -> None:
    (store / "8231.json").mkdir()

    code, payload = _json()

    assert code == 1
    assert payload["success"] is False
    assert payload["problems"][0]["port"] == 8231
