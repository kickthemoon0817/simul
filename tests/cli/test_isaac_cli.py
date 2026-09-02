"""CLI tests for Isaac bridge inspection and control commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from typer.testing import CliRunner

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult  # noqa: E402
from simul_mcp.cli import isaac as isaac_cli  # noqa: E402
from simul_mcp.cli.main import app  # noqa: E402


runner = CliRunner()


def _make_tools() -> SimpleNamespace:
    """Create a mocked IsaacTools-style namespace for CLI tests."""
    client = SimpleNamespace(
        address="127.0.0.1:8229",
        bridge_address="127.0.0.1:8229",
        bridge_enabled=True,
        bridge_request=AsyncMock(),
        execute_vscode_only=AsyncMock(),
    )
    return SimpleNamespace(_client=client)


def test_bridge_capabilities_json(monkeypatch) -> None:
    """The CLI should expose bridge permission state from capabilities."""
    tools = _make_tools()
    tools._client.bridge_request.return_value = {
        "status": "ok",
        "protocol_version": "1.0",
        "payload": {
            "transport": "simul_bridge",
            "allow_unsafe_execution": True,
            "actions": ["ping", "capabilities", "execute_script"],
        },
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-capabilities"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["allow_unsafe_execution"] is True
    assert payload["bridge_address"] == "127.0.0.1:8229"
    assert "execute_script" in payload["actions"]


def test_bridge_config_json(monkeypatch) -> None:
    """The CLI should read the running bridge config through VS Code."""
    tools = _make_tools()
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True,
        output=json.dumps(
            {
                "extension_enabled": True,
                "host": "127.0.0.1",
                "port": 8229,
                "allow_unsafe_execution": True,
                "max_request_bytes": 1048576,
                "max_response_bytes": 10485760,
            }
        ),
        transport="vscode",
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-config"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["extension_enabled"] is True
    assert payload["allow_unsafe_execution"] is True
    assert payload["port"] == 8229


def test_bridge_set_unsafe_json(monkeypatch) -> None:
    """The CLI should update unsafe execution and surface restart status."""
    tools = _make_tools()
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True,
        output=json.dumps(
            {
                "allow_unsafe_execution": False,
                "restart_requested": True,
                "restarted": True,
            }
        ),
        transport="vscode",
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-set-unsafe", "--disable"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["allow_unsafe_execution"] is False
    assert payload["restarted"] is True


# ---------------------------------------------------------------------------
# bridge-up — closes the iter8 UX gap (manual enable-extension workaround).
# ---------------------------------------------------------------------------


def _make_bridge_up_tools() -> SimpleNamespace:
    """Mock tools with the extra fields/methods bridge-up needs."""
    tools = _make_tools()
    tools._client.vscode_address = "127.0.0.1:8226"
    tools.enable_isaac_extension = AsyncMock()
    return tools


def test_bridge_up_already_reachable(monkeypatch) -> None:
    """First branch: bridge already responds → action=already-up, no enable call."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.return_value = {"status": "ok"}
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "already-up"
    assert payload["bridge_reachable"] is True
    assert payload["success"] is True
    tools.enable_isaac_extension.assert_not_called()


def test_bridge_up_isaac_not_running(monkeypatch) -> None:
    """Second branch: neither bridge nor VS Code reachable → exit non-zero
    with NotRunning error."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.side_effect = ConnectionRefusedError("nope")
    tools._client.execute_vscode_only.side_effect = ConnectionRefusedError("nope")
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code != 0
    assert "NotRunning" in result.stdout
    tools.enable_isaac_extension.assert_not_called()


def test_bridge_up_extension_enable_fails(monkeypatch) -> None:
    """Third branch: bridge down, VS Code up, but enable-extension fails →
    exit non-zero with ExtensionNotRegistered."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.side_effect = ConnectionRefusedError("nope")
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True, output="pong\n", transport="vscode"
    )
    tools.enable_isaac_extension.return_value = {
        "success": False,
        "error": "Extension not found: khemoo.simul.mcp",
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code != 0
    assert "ExtensionNotRegistered" in result.stdout
    tools.enable_isaac_extension.assert_awaited_once()


def test_bridge_up_auto_enables_then_reachable(monkeypatch) -> None:
    """Fourth branch (the win): bridge down, VS Code up, enable succeeds,
    bridge then becomes reachable on re-probe → action=auto-enabled,
    success=True. The whole point of the iter8 finding being closed.

    Iter10 review HIGH: the post-enable re-probe now retries up to 6
    times; this test gives it a couple of refusals before success to
    exercise the retry loop's middle case."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.side_effect = [
        ConnectionRefusedError("initial probe — bridge not enabled"),
        ConnectionRefusedError("retry 1 — port not bound yet"),
        {"status": "ok"},  # retry 2 succeeds
    ]
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True, output="pong\n", transport="vscode"
    )
    tools.enable_isaac_extension.return_value = {
        "success": True,
        "enabled": True,
        "extension_id": "khemoo.simul.mcp-0.0.31",
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)
    # Strip retry sleep for fast test — actual prod delay is 0.5 s.
    monkeypatch.setattr(
        isaac_cli.asyncio, "sleep", AsyncMock(return_value=None)
    )

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "auto-enabled"
    assert payload["bridge_reachable"] is True
    assert payload["extension_enabled"] is True
    assert payload["success"] is True
    tools.enable_isaac_extension.assert_awaited_once_with(
        extension_id="khemoo.simul.mcp"
    )


def _write_bridge_source(root: Path, version: str) -> Path:
    """Build a minimal bridge-ext source dir for install-bridge tests."""
    source = root / "exts" / "khemoo.simul.mcp"
    (source / "config").mkdir(parents=True, exist_ok=True)
    (source / "config" / "extension.toml").write_text(
        f'[package]\nversion = "{version}"\n', encoding="utf-8"
    )
    (source / "khemoo").mkdir(parents=True, exist_ok=True)
    (source / "khemoo" / "__init__.py").write_text("", encoding="utf-8")
    return source


def _make_isaac_root(root: Path) -> Path:
    isaac_root = root / "isaac-sim"
    (isaac_root / "extsUser").mkdir(parents=True, exist_ok=True)
    return isaac_root


# ---------------------------------------------------------------------------
# install-bridge — closes the iter11 publish gap (Isaac loads from extsUser,
# repo bumps don't propagate without an explicit copy/symlink).
# ---------------------------------------------------------------------------


def test_install_bridge_refuses_without_isaac_root(tmp_path: Path, monkeypatch) -> None:
    """No --isaac-root and no $ISAAC_SIM_PATH → InvalidArgument exit."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)

    result = runner.invoke(
        app,
        ["--json", "isaac", "install-bridge", "--source", str(source)],
    )

    assert result.exit_code != 0
    assert "InvalidArgument" in result.stdout
    assert "ISAAC_SIM_PATH" in result.stdout


def test_install_bridge_refuses_when_extsUser_missing(tmp_path: Path) -> None:
    """isaac-root that's not actually an Isaac install (no extsUser dir) →
    refuses BEFORE writing anything."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    fake_isaac = tmp_path / "not-actually-isaac"
    fake_isaac.mkdir()

    result = runner.invoke(
        app,
        [
            "--json", "isaac", "install-bridge",
            "--isaac-root", str(fake_isaac),
            "--source", str(source),
        ],
    )

    assert result.exit_code != 0
    assert "extsUser" in result.stdout


def test_install_bridge_copies_into_extsUser(tmp_path: Path) -> None:
    """Fresh install: dest doesn't exist → action=copied, version matches."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    result = runner.invoke(
        app,
        [
            "--json", "isaac", "install-bridge",
            "--isaac-root", str(isaac_root),
            "--source", str(source),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert payload["version"] == "0.0.33"
    assert payload["previous_version"] is None
    assert payload["success"] is True
    # File system: dest exists with correct version.
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert (dest / "config" / "extension.toml").is_file()
    assert (dest / "khemoo" / "__init__.py").is_file()


def test_install_bridge_already_current_no_op(tmp_path: Path) -> None:
    """Re-running with the same version is a clean no-op (action=already-current).
    No --force, no copy, no error."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    # First install
    runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])
    dest_toml = isaac_root / "extsUser" / "khemoo.simul.mcp" / "config" / "extension.toml"
    mtime_before = dest_toml.stat().st_mtime

    # Re-run
    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "already-current"
    assert payload["version"] == "0.0.33"
    # File untouched.
    assert dest_toml.stat().st_mtime == mtime_before


def test_install_bridge_replaces_stale_dest(tmp_path: Path) -> None:
    """Pre-existing stale dest (e.g. iter11's verifier scenario: dest at
    0.0.13, source at 0.0.33) → action=copied, previous_version captured,
    new version verified."""
    isaac_root = _make_isaac_root(tmp_path)
    # Pre-populate stale dest at 0.0.13.
    stale_dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    (stale_dest / "config").mkdir(parents=True)
    (stale_dest / "config" / "extension.toml").write_text(
        '[package]\nversion = "0.0.13"\n', encoding="utf-8"
    )
    # New source at 0.0.33.
    source = _write_bridge_source(tmp_path, "0.0.33")

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert payload["previous_version"] == "0.0.13"
    assert payload["version"] == "0.0.33"
    assert payload["success"] is True


def test_install_bridge_symlink_mode(tmp_path: Path) -> None:
    """--symlink uses ln -s instead of copy. Repo edits then propagate
    automatically (no re-install needed)."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
        "--symlink",
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "symlinked"
    assert payload["version"] == "0.0.33"
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert dest.is_symlink()
    assert dest.resolve() == source


def test_install_bridge_auto_discovers_source_via_walking_parents(
    tmp_path: Path, monkeypatch
) -> None:
    """Legacy fallback path: when the bundled bridge_ext/ sibling is
    absent (very old editable checkouts), the command walks parents
    from the simul_mcp package's __file__ looking for
    exts/khemoo.simul.mcp/. iter14 made this the second-choice path
    after the bundled location; this test exercises it by pointing
    __file__ at a fake package with no bridge_ext sibling."""
    # Build a fake "repo" layout with the bridge source under exts/.
    fake_repo = tmp_path / "fake_repo"
    fake_pkg = fake_repo / "src" / "simul_mcp"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    # NOTE: deliberately no bridge_ext/ sibling — forces the bundled
    # check to miss so the legacy walk path runs.
    _write_bridge_source(fake_repo, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    # Point simul_mcp.__file__ at the fake package so the walk finds the
    # fake repo's exts dir instead of the real one.
    import simul_mcp as _sm
    monkeypatch.setattr(_sm, "__file__", str(fake_pkg / "__init__.py"))

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        # NOTE: no --source — exercising the auto-discovery walk
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert payload["version"] == "0.0.33"
    # Source path resolution went through our fake repo, not the real one.
    assert str(fake_repo) in payload["source"]


def test_install_bridge_prefers_bundled_source_over_legacy_walk(
    tmp_path: Path, monkeypatch
) -> None:
    """iter14 contract: the bundled copy at
    ``simul_mcp/bridge_ext/khemoo.simul.mcp/`` is the primary source.
    A pip-installed user has no ``exts/`` dir anywhere on disk; the
    bundled copy must resolve without any repo layout present.

    This test points ``simul_mcp.__file__`` at a fake package that has
    a ``bridge_ext/khemoo.simul.mcp/`` sibling but NO ``exts/`` parent
    chain — exactly the wheel-install topology — and verifies the
    install succeeds from the bundled source.
    """
    # Fake wheel-install layout: simul_mcp/__init__.py with
    # bridge_ext/khemoo.simul.mcp/ as a sibling, no exts/ anywhere.
    fake_pkg = tmp_path / "wheel_pkg" / "simul_mcp"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    bundled = fake_pkg / "bridge_ext" / "khemoo.simul.mcp"
    (bundled / "config").mkdir(parents=True)
    (bundled / "config" / "extension.toml").write_text(
        '[package]\nversion = "0.0.99"\n', encoding="utf-8"
    )
    isaac_root = _make_isaac_root(tmp_path)

    import simul_mcp as _sm
    monkeypatch.setattr(_sm, "__file__", str(fake_pkg / "__init__.py"))

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert payload["version"] == "0.0.99"
    # Pin the exact resolved path — substring-only checks would pass
    # for any path that happens to contain "bridge_ext" elsewhere.
    assert payload["source"] == str(bundled)


def test_install_bridge_bundled_source_with_symlink_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """iter14: bundled-source resolution must compose cleanly with
    --symlink. A regression that mutated source_p between the bundled
    resolve and the dest.symlink_to call would only show up when both
    code paths run together — neither single-mode test catches it.
    """
    fake_pkg = tmp_path / "wheel_pkg" / "simul_mcp"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    bundled = fake_pkg / "bridge_ext" / "khemoo.simul.mcp"
    (bundled / "config").mkdir(parents=True)
    (bundled / "config" / "extension.toml").write_text(
        '[package]\nversion = "0.0.99"\n', encoding="utf-8"
    )
    isaac_root = _make_isaac_root(tmp_path)

    import simul_mcp as _sm
    monkeypatch.setattr(_sm, "__file__", str(fake_pkg / "__init__.py"))

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--symlink",
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "symlinked"
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert dest.is_symlink()
    # The symlink target must be the exact bundled path that the
    # resolver picked — not something silently mutated mid-flow.
    assert dest.resolve() == bundled.resolve()


def test_install_bridge_bundled_toml_missing_falls_through(
    tmp_path: Path, monkeypatch
) -> None:
    """Defensive: when the bundled directory exists but
    config/extension.toml is missing (corrupted install / partial
    wheel extraction), the resolver must skip the bundled candidate
    and either find a legacy fallback or exit cleanly with
    SourceNotFound — never crash on the missing file.
    """
    fake_pkg = tmp_path / "wheel_pkg" / "simul_mcp"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    # Bundled dir exists but the toml is absent → must not be picked.
    bundled_dir = fake_pkg / "bridge_ext" / "khemoo.simul.mcp" / "config"
    bundled_dir.mkdir(parents=True)
    isaac_root = _make_isaac_root(tmp_path)

    import simul_mcp as _sm
    monkeypatch.setattr(_sm, "__file__", str(fake_pkg / "__init__.py"))

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
    ])

    assert result.exit_code != 0
    assert "SourceNotFound" in result.stdout


def test_install_bridge_picks_up_isaac_sim_path_env_var(
    tmp_path: Path, monkeypatch
) -> None:
    """Test-engineer Gap (env-var resolution): with --isaac-root absent
    AND $ISAAC_SIM_PATH set, the command must use the env var. Pre-fix
    this code path was only negatively tested (env unset = error)."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)
    monkeypatch.setenv("ISAAC_SIM_PATH", str(isaac_root))

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        # NOTE: no --isaac-root — exercising the env var fallback
        "--source", str(source),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert str(isaac_root) in payload["dest"]


def test_install_bridge_unreadable_dest_toml_falls_through_cleanly(
    tmp_path: Path,
) -> None:
    """Code-reviewer LOW: a partial-extract / corrupted prior-install
    leaves dest/config/extension.toml as a non-toml or unreadable file.
    _read_version returns None (no uncaught PermissionError or
    UnicodeDecodeError), the version mismatch triggers replace, and
    the new version is verified — the install is recoverable, not
    crash-prone."""
    isaac_root = _make_isaac_root(tmp_path)
    # Simulate a partial extraction: dest exists but the toml is
    # unreadable garbage bytes (definitely not valid UTF-8).
    bad_dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    (bad_dest / "config").mkdir(parents=True)
    (bad_dest / "config" / "extension.toml").write_bytes(b"\xff\xfe\x00\x00\x80\x81")

    source = _write_bridge_source(tmp_path, "0.0.33")
    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    # Couldn't read the prior version — that's fine, treated as unknown.
    assert payload["previous_version"] is None
    assert payload["version"] == "0.0.33"


def test_install_bridge_force_replaces_matching_version(tmp_path: Path) -> None:
    """--force replaces dest even when versions match (e.g. mode change
    from copy to symlink, or stale-but-same-version content)."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)
    runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    # Force re-install as symlink.
    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
        "--force", "--symlink",
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "symlinked"
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert dest.is_symlink()


def test_bridge_up_enable_succeeds_but_bridge_stays_down(monkeypatch) -> None:
    """Fifth branch (test-engineer's flagged gap): enable succeeded but
    bridge port still doesn't bind even after the retry loop
    exhausts. action=auto-enabled but success=False, exit non-zero."""
    tools = _make_bridge_up_tools()
    # All 7 probes (1 initial + 6 retries) refuse.
    tools._client.bridge_request.side_effect = ConnectionRefusedError(
        "still not up"
    )
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True, output="pong\n", transport="vscode"
    )
    tools.enable_isaac_extension.return_value = {
        "success": True,
        "enabled": True,
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)
    monkeypatch.setattr(
        isaac_cli.asyncio, "sleep", AsyncMock(return_value=None)
    )

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "auto-enabled"
    assert payload["bridge_reachable"] is False
    assert payload["success"] is False
    assert payload["extension_enabled"] is True


def test_capture_requests_inline_and_writes_base64(monkeypatch, tmp_path) -> None:
    """A capture under the inline cap arrives as base64 and lands on disk."""
    import base64 as b64

    png = b"\x89PNG-fake-bytes"
    tools = _make_tools()
    tools.capture_isaac_viewport = AsyncMock(
        return_value={
            "path": "/tmp/simul_capture_ab.png",
            "width": 320,
            "height": 180,
            "format": "png",
            "size_bytes": len(png),
            "image_base64": b64.b64encode(png).decode("ascii"),
            "success": True,
        }
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    out = tmp_path / "cap.png"
    result = runner.invoke(app, ["--json", "isaac", "capture", str(out)])

    assert result.exit_code == 0
    assert tools.capture_isaac_viewport.call_args.kwargs["inline"] is True
    assert out.read_bytes() == png
    payload = json.loads(result.stdout)
    assert payload["file_path"] == str(out.resolve())
    assert "image_base64" not in payload


def test_capture_falls_back_to_host_path_copy(monkeypatch, tmp_path) -> None:
    """Above the inline cap the tool returns only a path; the CLI copies it.

    The CLI previously required ``image_base64`` and reported "no image data"
    for every path-only response, which is the tool's default shape.
    """
    png = b"\x89PNG-big-fake-bytes"
    host_file = tmp_path / "simul_capture_cd.png"
    host_file.write_bytes(png)

    tools = _make_tools()
    tools.capture_isaac_viewport = AsyncMock(
        return_value={
            "path": str(host_file),
            "width": 1920,
            "height": 1080,
            "format": "png",
            "size_bytes": len(png),
            "inline_skipped": "above cap",
            "success": True,
        }
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    out = tmp_path / "out" / "cap.png"
    result = runner.invoke(app, ["--json", "isaac", "capture", str(out)])

    assert result.exit_code == 0
    assert out.read_bytes() == png
    payload = json.loads(result.stdout)
    assert payload["file_path"] == str(out.resolve())


def test_capture_reports_missing_image_with_detail(monkeypatch, tmp_path) -> None:
    """No base64 and no readable path is a hard error naming the reason."""
    tools = _make_tools()
    tools.capture_isaac_viewport = AsyncMock(
        return_value={
            "path": str(tmp_path / "does-not-exist.png"),
            "inline_skipped": "above cap",
            "success": True,
        }
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(
        app, ["--json", "isaac", "capture", str(tmp_path / "cap.png")]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "CaptureError"
    assert "above cap" in payload["error"]


# ---------------------------------------------------------------------------
# launch --dry-run: version-aware transport extension selection
# ---------------------------------------------------------------------------
def _write_isaac_root(tmp_path: Path, version: str, *, with_bridge: bool) -> Path:
    root = tmp_path / f"isaac-sim-{version}"
    root.mkdir()
    (root / "VERSION").write_text(f"{version}-rc.1+release.1.abc.gl\n")
    (root / "isaac-sim.sh").write_text("#!/bin/sh\n")
    exts_user = root / "extsUser"
    exts_user.mkdir()
    if with_bridge:
        (exts_user / "khemoo.simul.mcp" / "config").mkdir(parents=True)
        (exts_user / "khemoo.simul.mcp" / "config" / "extension.toml").write_text(
            '[package]\nversion = "0.1.0"\n'
        )
    return root


def test_launch_dry_run_isaac_six_enables_python_server(tmp_path: Path) -> None:
    root = _write_isaac_root(tmp_path, "6.0.1", with_bridge=True)

    result = runner.invoke(app, ["--json", "isaac", "launch", "--isaac-root", str(root), "--dry-run"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["version"] == "6.0.1"
    assert payload["transport_extension"] == "isaacsim.code_editor.python_server"
    assert payload["bridge_extension_present"] is True
    command = payload["command"]
    assert command[0] == str(root / "isaac-sim.sh")
    assert command[1:3] == ["--enable", "isaacsim.code_editor.python_server"]
    assert "--/exts/isaacsim.code_editor.python_server/port=8226" in command
    assert "khemoo.simul.mcp" in command
    assert "--/exts/khemoo.simul.mcp/port=8229" in command
    assert "--no-window" in command


def test_launch_dry_run_isaac_five_enables_vscode(tmp_path: Path, monkeypatch) -> None:
    root = _write_isaac_root(tmp_path, "5.1.0", with_bridge=False)
    monkeypatch.setenv("ISAAC_SIM_PATH", str(root))

    result = runner.invoke(
        app,
        ["--json", "isaac", "launch", "--dry-run", "--no-headless", "--socket-port", "8300", "--kit-arg", "--verbose"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["transport_extension"] == "isaacsim.code_editor.vscode"
    assert payload["bridge_extension_present"] is False
    assert "install-bridge" in payload["hint"]
    command = payload["command"]
    assert "--/exts/isaacsim.code_editor.vscode/port=8300" in command
    assert "khemoo.simul.mcp" not in command
    assert "--no-window" not in command
    assert command[-1] == "--verbose"


def test_launch_auth_token_requires_isaac_six(tmp_path: Path) -> None:
    root = _write_isaac_root(tmp_path, "5.1.0", with_bridge=True)

    result = runner.invoke(
        app, ["--json", "isaac", "launch", "--isaac-root", str(root), "--dry-run", "--auth-token", "t"]
    )

    assert result.exit_code != 0
    assert "InvalidArgument" in result.stdout


def test_launch_auth_token_configures_python_server(tmp_path: Path) -> None:
    root = _write_isaac_root(tmp_path, "6.0.0", with_bridge=True)

    result = runner.invoke(
        app, ["--json", "isaac", "launch", "--isaac-root", str(root), "--dry-run", "--auth-token", "t0k"]
    )

    assert result.exit_code == 0, result.stdout
    command = json.loads(result.stdout)["command"]
    assert "--/exts/isaacsim.code_editor.python_server/require_auth=true" in command
    assert "--/exts/isaacsim.code_editor.python_server/auth_token=t0k" in command


def test_launch_rejects_root_without_version_file(tmp_path: Path) -> None:
    root = tmp_path / "not-isaac"
    root.mkdir()
    (root / "isaac-sim.sh").write_text("#!/bin/sh\n")

    result = runner.invoke(app, ["--json", "isaac", "launch", "--isaac-root", str(root), "--dry-run"])

    assert result.exit_code != 0
    assert "UnsupportedInstall" in result.stdout
