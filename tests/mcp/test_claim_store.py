"""The claim store every server process on a machine shares.

Claims live in one JSON file per instance port. The file was read and
rewritten with no lock and an ``O_TRUNC`` write, so concurrent claims from
several server processes overwrote each other (219 of 240 lost in an audit),
a reader saw a truncated file as "no claims" while another process rewrote it,
and a corrupt file read as "no claims", so ``enforce_claims`` failed open.
These tests drive real processes against one store, and the repair and
fail-closed rules a damaged or unreadable store follows.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from simul.mcp import session_manager as session_manager_module
from simul.mcp.session_manager import (
    ClaimStoreError,
    ClaimView,
    InstanceSession,
    SessionManager,
    claim_is_orphaned,
)

SRC = Path(__file__).resolve().parents[2] / "src"


def _spawn(script: str, *args: str) -> subprocess.Popen:
    """Start a Python worker running ``script`` against this checkout's source."""
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.Popen(
        [sys.executable, "-c", script, *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# Waits for the go file, then registers its agent: every worker starts its
# read-modify-write within the same few milliseconds.
_CLAIMER = """
import os, sys, time
from pathlib import Path
from simul.mcp.session_manager import InstanceSession
store, go, agent, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
while not go.exists():
    time.sleep(0.0005)
session = InstanceSession(8226, store)
if mode == "exclusive":
    conflict = lambda view: view.sessions[0] if view.sessions else None
else:
    conflict = None
print(session.register(agent, "purpose " + agent, owner="owner-" + agent, conflict=conflict)["status"])
"""

_HEARTBEATER = """
import sys, time
from pathlib import Path
from simul.mcp.session_manager import InstanceSession
store, seconds = Path(sys.argv[1]), float(sys.argv[2])
session = InstanceSession(8226, store)
deadline = time.monotonic() + seconds
while time.monotonic() < deadline:
    session.heartbeat("holder", "tool")
"""


def _race(store: Path, workers: int, mode: str) -> List[str]:
    go = store / "go"
    processes = [_spawn(_CLAIMER, str(store), str(go), f"agent-{i}", mode) for i in range(workers)]
    time.sleep(0.5)  # let every interpreter start and reach its wait loop
    go.touch()
    outputs = []
    for process in processes:
        out, err = process.communicate(timeout=60)
        assert process.returncode == 0, err
        outputs.append(out.strip())
    return outputs


def test_concurrent_claims_from_eight_processes_are_all_kept(tmp_path: Path) -> None:
    for trial in range(3):
        store = tmp_path / f"trial-{trial}"
        store.mkdir()

        assert _race(store, 8, "advisory") == ["registered"] * 8

        document = json.loads((store / "8226.json").read_text())
        assert sorted(s["agent_id"] for s in document["sessions"]) == [f"agent-{i}" for i in range(8)]
        assert not list(store.glob(".*.tmp")), "no temporary file is left behind"


def test_an_enforced_claim_race_has_exactly_one_winner(tmp_path: Path) -> None:
    """The conflict check runs under the same lock as the write.

    Checking for a holder and then registering under separate locks let two
    processes both see a free instance and both claim it.
    """
    outcomes = _race(tmp_path, 8, "exclusive")

    assert sorted(outcomes) == ["refused"] * 7 + ["registered"]
    assert len(InstanceSession(8226, tmp_path).view().sessions) == 1


def test_a_reader_never_misses_a_claim_while_other_processes_heartbeat_it(tmp_path: Path) -> None:
    InstanceSession(8226, tmp_path).register("holder", "hold", owner="o")
    heartbeaters = [_spawn(_HEARTBEATER, str(tmp_path), "1.5") for _ in range(2)]
    reader = InstanceSession(8226, tmp_path)
    reads = misses = 0
    try:
        while any(p.poll() is None for p in heartbeaters):
            reads += 1
            if not any(s["agent_id"] == "holder" for s in reader.view().sessions):
                misses += 1
    finally:
        for process in heartbeaters:
            _, err = process.communicate(timeout=60)
            assert process.returncode == 0, err

    assert reads > 50
    assert misses == 0
    assert json.loads((tmp_path / "8226.json").read_text())["sessions"][0]["agent_id"] == "holder"


def test_a_record_file_with_trailing_garbage_keeps_its_first_record_set(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An interleaved write leaves one complete document followed by debris."""
    store = InstanceSession(8226, tmp_path)
    store.register("holder", "hold", owner="o")
    path = tmp_path / "8226.json"
    path.write_text(path.read_text() + '\n  }\n]}\n')

    with caplog.at_level(logging.WARNING, logger="simul.mcp.session_manager"):
        view = store.view()

    assert [s["agent_id"] for s in view.sessions] == ["holder"]
    assert view.repaired_at is not None
    assert "were damaged" in caplog.text
    repaired = json.loads(path.read_text())
    assert repaired["repaired_at"] == view.repaired_at
    assert [s["agent_id"] for s in repaired["sessions"]] == ["holder"]


def test_an_unparseable_record_file_is_moved_aside_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "8226.json").write_text('{"port": 8226, "sessions": [{"agent_id": ')

    with caplog.at_level(logging.WARNING, logger="simul.mcp.session_manager"):
        status = InstanceSession(8226, tmp_path).get_status()

    (quarantined,) = tmp_path.glob("8226.json.corrupt-*")
    assert quarantined.read_text().startswith('{"port": 8226')
    assert status["sessions"] == []
    assert status["status"] == "unverified"
    assert "repaired_at" in status
    assert "moved to" in caplog.text


def test_a_repair_window_ends_after_one_claim_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "8226.json").write_text("not json")
    monkeypatch.setattr(session_manager_module, "CLAIM_TTL_SECONDS", 0.05)
    assert InstanceSession(8226, tmp_path).view().repaired_at is not None

    time.sleep(0.1)

    assert InstanceSession(8226, tmp_path).view().repaired_at is None
    assert not (tmp_path / "8226.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_an_unwritable_store_raises_instead_of_reporting_a_claim(tmp_path: Path) -> None:
    store_dir = tmp_path / "sessions"
    store_dir.mkdir()
    store_dir.chmod(0o500)
    try:
        with pytest.raises(ClaimStoreError):
            InstanceSession(8226, store_dir).register("agent", "purpose", owner="o")
        # A read of a store nobody could write reports nothing claimed.
        assert InstanceSession(8226, store_dir).view().sessions == []
    finally:
        store_dir.chmod(0o700)


def test_the_session_manager_does_not_create_its_directory_until_it_writes(tmp_path: Path) -> None:
    """A home directory that cannot be written must not stop the server starting."""
    manager = SessionManager(tmp_path / "sessions")

    assert manager.get_all_sessions() == []
    assert not (tmp_path / "sessions").exists()

    manager.get_instance_session(8226).register("agent", "purpose")
    assert (tmp_path / "sessions" / "8226.json").exists()


def test_records_name_their_process_so_an_exited_server_reads_as_orphaned(tmp_path: Path) -> None:
    InstanceSession(8226, tmp_path).register("live", "purpose", owner="o")
    exited = _spawn(
        "import sys\nfrom pathlib import Path\n"
        "from simul.mcp.session_manager import InstanceSession\n"
        "InstanceSession(8226, Path(sys.argv[1])).register('gone', 'purpose', owner='p')\n",
        str(tmp_path),
    )
    exited.communicate(timeout=60)

    records = {s["agent_id"]: s for s in InstanceSession(8226, tmp_path).view().sessions}
    assert records["live"]["pid"] == os.getpid()
    assert claim_is_orphaned(records["live"]) is False
    if sys.platform != "win32":
        assert claim_is_orphaned(records["gone"]) is True


def test_marking_an_owner_orphaned_and_clearing(tmp_path: Path) -> None:
    store = InstanceSession(8226, tmp_path)
    store.register("a", "one", owner="o1")
    store.register("b", "two", owner="o2")

    assert store.mark_orphaned("o1") == 1
    records = {s["agent_id"]: s for s in store.view().sessions}
    assert records["a"]["orphaned"] is True and "orphaned" not in records["b"]

    # Claiming again clears the mark.
    store.register("a", "one", owner="o3")
    assert "orphaned" not in {s["agent_id"]: s for s in store.view().sessions}["a"]

    manager = SessionManager(tmp_path)
    InstanceSession(8230, tmp_path).register("c", "three")
    assert [(r["port"], r["agent_id"]) for r in manager.clear(agent_id="b")] == [(8226, "b")]
    assert sorted((r["port"], r["agent_id"]) for r in manager.clear()) == [(8226, "a"), (8230, "c")]
    assert manager.get_all_sessions() == []


def test_a_conflict_sees_the_repair_window(tmp_path: Path) -> None:
    (tmp_path / "8226.json").write_text("garbage")
    seen: List[ClaimView] = []

    def conflict(view: ClaimView) -> Dict[str, Any] | None:
        seen.append(view)
        return {"agent_id": None} if view.repaired_at is not None else None

    result = InstanceSession(8226, tmp_path).register("a", "p", owner="o", conflict=conflict)

    assert result["status"] == "refused"
    assert seen and seen[0].repaired_at is not None
