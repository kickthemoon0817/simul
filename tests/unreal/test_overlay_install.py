"""Installation must preserve projects when compilation fails or plugins differ."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simul_mcp.adapters import unreal_overlay
from simul_mcp.adapters.unreal_setup import patch_uproject


@pytest.fixture
def project(tmp_path, monkeypatch):
    engine = tmp_path / "EngineRoot"
    build = engine / "Engine/Build"
    (build / "BatchFiles").mkdir(parents=True)
    (build / "BatchFiles/RunUAT.sh").touch()
    (build / "Build.version").write_text('{"version": "test"}')
    project = tmp_path / "Project/Test.uproject"
    project.parent.mkdir()
    project.write_text("{}")
    monkeypatch.setattr(unreal_overlay.platform, "system", lambda: "Darwin")
    return project, engine


def test_build_failure_does_not_install_partial_plugin(project, monkeypatch):
    uproject, engine = project
    monkeypatch.setattr(
        unreal_overlay.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=6)
    )
    with pytest.raises(RuntimeError, match="build failed"):
        unreal_overlay.install_agent_overlay(uproject, engine)
    assert not (uproject.parent / "Plugins/SimulAgentOverlay").exists()
    assert uproject.read_text() == "{}"


def test_install_is_idempotent_and_refuses_replacing_existing_sources(
    project, monkeypatch
):
    uproject, engine = project
    calls = []

    def build(command, **kwargs):
        calls.append(command)
        package = Path(
            next(
                s.removeprefix("-Package=")
                for s in command
                if s.startswith("-Package=")
            )
        )
        (package / "Binaries/Mac").mkdir(parents=True)
        (package / "Binaries/Mac/UnrealEditor-SimulAgentOverlay.dylib").write_bytes(
            b"test"
        )
        (package / "SimulAgentOverlay.uplugin").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(unreal_overlay.subprocess, "run", build)
    result = unreal_overlay.install_agent_overlay(uproject, engine)
    assert result["changed"] and len(calls) == 1
    assert not unreal_overlay.install_agent_overlay(uproject, engine)["changed"]
    assert len(calls) == 1
    receipt = Path(result["path"]) / ".simul-build.json"
    receipt.unlink()
    with pytest.raises(ValueError, match="Existing overlay differs"):
        unreal_overlay.install_agent_overlay(uproject, engine)
    assert (Path(result["path"]) / "SimulAgentOverlay.uplugin").read_text() == "{}"


def test_setup_enables_previously_disabled_overlay(project):
    uproject, _ = project
    uproject.write_text(
        json.dumps({"Plugins": [{"Name": "SimulAgentOverlay", "Enabled": False}]})
    )
    assert patch_uproject(uproject, agent_overlay=True).changed
    assert not patch_uproject(uproject, agent_overlay=True).changed
    assert all(p["Enabled"] for p in json.loads(uproject.read_text())["Plugins"])


def test_overlay_auto_detection_bypasses_macos_launchservices(project, monkeypatch):
    from simul_mcp.adapters import unreal_setup

    uproject, engine = project
    binary = engine / "Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
    monkeypatch.setattr(unreal_setup, "_macos_macos_binary", lambda root: binary)
    monkeypatch.setattr(unreal_setup.shutil, "which", lambda name: "/usr/bin/open")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "open":
            return SimpleNamespace(returncode=0)  # LaunchServices knows Unreal.
        package = Path(
            next(
                s.removeprefix("-Package=")
                for s in command
                if s.startswith("-Package=")
            )
        )
        (package / "Binaries/Mac").mkdir(parents=True)
        (package / "SimulAgentOverlay.uplugin").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(unreal_overlay.subprocess, "run", run)
    result = unreal_overlay.install_agent_overlay(uproject, None)
    assert result["changed"]
    assert len(commands) == 1
    assert commands[0][0] == str(engine / "Engine/Build/BatchFiles/RunUAT.sh")
