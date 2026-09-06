"""Data files shipped inside the package and the sandbox roots derived from them.

``skills.md``, the API docs and the default YAML are read through
``importlib.resources`` so a wheel install serves them exactly like an editable
checkout. The sandbox allowlist's relative entries only exist in a checkout and
must be dropped, not pointed at ``site-packages``, when there is none.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import simul_mcp.utils.paths as paths_module
from simul_mcp.resources import find_checkout_root, resource, resource_filesystem_path
from simul_mcp.utils.paths import PathPolicy

_API_DOCS = ("core", "sensors", "physics", "replicator", "robots", "rendering", "assets")


def test_skills_document_is_packaged() -> None:
    skills = resource("skills.md")

    assert skills.is_file()
    assert "execute_isaac_script" in skills.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", _API_DOCS)
def test_api_reference_docs_are_packaged(name: str) -> None:
    document = resource("docs", "api", f"{name}.md")

    assert document.is_file()
    assert document.read_text(encoding="utf-8").strip()


def test_default_and_logging_yaml_are_packaged() -> None:
    assert resource_filesystem_path("config", "default.yaml").is_file()
    assert resource_filesystem_path("config", "logging.yaml").is_file()


def test_repo_root_no_longer_holds_the_moved_files() -> None:
    repo = Path(__file__).resolve().parents[2]

    assert not (repo / "skills.md").exists()
    assert not (repo / "docs" / "api").exists()
    assert not (repo / "config" / "isaac" / "default.yaml").exists()
    assert not (repo / "config" / "blender").exists()


def test_checkout_root_is_detected_for_the_source_tree() -> None:
    root = find_checkout_root()

    assert root is not None
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "simul_mcp" / "resources" / "skills.md").is_file()


def test_relative_allowlist_entries_resolve_against_the_checkout(tmp_path: Path) -> None:
    (tmp_path / "examples").mkdir()
    policy = PathPolicy(enabled=True, allowed_paths=["examples", "/tmp/simul_mcp"], project_root=tmp_path)

    assert policy.allowed_roots == [(tmp_path / "examples").resolve(), Path("/tmp/simul_mcp").resolve()]
    assert policy.is_allowed("examples/scene.usd")
    assert not policy.is_allowed("/etc/shadow")


def test_relative_allowlist_entries_are_dropped_without_a_checkout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(paths_module, "find_checkout_root", lambda: None)
    # setup_logging in other tests turns propagation off for simul_mcp loggers,
    # so capture at the emitting logger rather than at the root.
    paths_logger = logging.getLogger("simul_mcp.utils.paths")
    paths_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="simul_mcp.utils.paths"):
            policy = PathPolicy(enabled=True, allowed_paths=["examples", "tests/data", "/tmp/simul_mcp"])
    finally:
        paths_logger.removeHandler(caplog.handler)

    assert policy.allowed_roots == [Path("/tmp/simul_mcp").resolve()]
    dropped = [record.getMessage() for record in caplog.records if "Dropping relative sandbox path" in record.getMessage()]
    assert len(dropped) == 2
    assert not any("lib/python" in str(root) for root in policy.allowed_roots)


def test_relative_tool_paths_resolve_against_cwd_without_a_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths_module, "find_checkout_root", lambda: None)
    monkeypatch.chdir(tmp_path)
    policy = PathPolicy(enabled=True, allowed_paths=[str(tmp_path)])

    assert policy.resolve("scene.usd") == (tmp_path / "scene.usd").resolve()
    assert policy.is_allowed("scene.usd")
