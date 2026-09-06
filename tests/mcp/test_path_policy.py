"""Unit tests for the sandbox path policy.

Three contracts live here. A string shaped like ``<scheme>://`` is a URL and is
judged by the scheme allowlist, never resolved as a project-relative path.
``file://`` URLs are the exception: they name a local file, so they are
converted and checked like one. And whatever passes the check must be embedded
downstream in its *authorized* form — the resolved absolute path or the
verbatim URL — because the receiving application resolves relative paths
against a different working directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


from simul_mcp.config import Settings
from simul_mcp.utils.paths import PathPolicy, SandboxDenied


def _policy(tmp_path: Path, **overrides: object) -> PathPolicy:
    root = tmp_path / "sandbox"
    root.mkdir(exist_ok=True)
    kwargs: dict = {
        "enabled": True,
        "allowed_paths": [str(root), "relative_root"],
        "project_root": tmp_path,
    }
    kwargs.update(overrides)
    return PathPolicy(**kwargs)


class TestUrlClassification:
    @pytest.mark.parametrize(
        ("value", "scheme"),
        [
            ("omniverse://nucleus/Projects/a.usd", "omniverse"),
            ("OMNIVERSE://nucleus/a.usd", "omniverse"),
            ("https://example.com/a.usd", "https"),
            ("file:///tmp/a.usd", "file"),
            ("s3+https://bucket/a.usd", "s3+https"),
            ("/tmp/a.usd", None),
            ("relative/a.usd", None),
            ("C:\\\\Users\\\\a.usd", None),
            ("", None),
            ("://nothing", None),
        ],
    )
    def test_url_scheme(self, value: str, scheme: str | None) -> None:
        assert PathPolicy.url_scheme(value) == scheme

    def test_url_is_never_treated_as_project_relative(self, tmp_path: Path) -> None:
        """Before, ``omniverse://x`` resolved to ``<project>/omniverse:/x`` and failed containment."""
        policy = _policy(tmp_path, allowed_url_schemes=())
        with pytest.raises(ValueError):
            policy.resolve("omniverse://nucleus/a.usd")
        details = policy.denial_details("omniverse://nucleus/a.usd")
        assert details["file_path"] == "omniverse://nucleus/a.usd"

    def test_default_read_allowlist_admits_omniverse_only(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        url = "omniverse://nucleus/Projects/a.usd"
        assert policy.authorize(url) == url
        assert policy.is_allowed(url)
        assert not policy.is_allowed("https://example.com/a.usd")
        assert not policy.is_allowed("http://example.com/a.usd")

    def test_writes_to_urls_are_refused_by_default(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        url = "omniverse://nucleus/Projects/a.usd"
        assert policy.is_allowed(url)
        assert not policy.is_allowed(url, write=True)
        with pytest.raises(SandboxDenied) as excinfo:
            policy.authorize(url, write=True)
        assert excinfo.value.details["access"] == "write"
        assert excinfo.value.details["allowed_url_schemes"] == []

    def test_write_schemes_can_be_opted_in(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path, allowed_write_url_schemes=("omniverse",))
        url = "omniverse://nucleus/Projects/a.usd"
        assert policy.authorize(url, write=True) == url

    def test_scheme_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path, allowed_url_schemes=("HTTPS",))
        assert policy.is_allowed("https://example.com/a.usd")
        assert policy.is_allowed("HTTPS://example.com/a.usd")


class TestFileUrls:
    def test_file_url_inside_sandbox_becomes_a_local_path(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        target = tmp_path / "sandbox" / "scene.usd"
        authorized = policy.authorize(target.as_uri())
        assert authorized == str(target.resolve())
        assert PathPolicy.url_scheme(authorized) is None

    def test_file_url_outside_sandbox_is_refused(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        outside = (tmp_path / "elsewhere" / "scene.usd").as_uri()
        assert not policy.is_allowed(outside)

    def test_file_url_with_percent_encoding_is_decoded(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        target = tmp_path / "sandbox" / "my scene.usd"
        assert policy.authorize(target.as_uri()) == str(target.resolve())

    def test_file_url_naming_a_remote_host_is_refused(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        assert not policy.is_allowed("file://fileserver/share/scene.usd")
        with pytest.raises(SandboxDenied):
            policy.authorize("file://fileserver/share/scene.usd")

    def test_file_url_never_reaches_the_scheme_allowlist(self, tmp_path: Path) -> None:
        """Allowing ``file`` as a URL scheme must not bypass the root check."""
        policy = _policy(tmp_path, allowed_url_schemes=("file",))
        outside = (tmp_path / "elsewhere" / "scene.usd").as_uri()
        assert not policy.is_allowed(outside)


class TestAuthorizeContract:
    def test_relative_path_comes_back_absolute_under_project_root(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        authorized = policy.authorize("relative_root/scene.usd")
        assert authorized == str((tmp_path / "relative_root" / "scene.usd").resolve())
        assert Path(authorized).is_absolute()

    def test_tilde_and_env_vars_are_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIMUL_TEST_ROOT", str(tmp_path / "sandbox"))
        policy = _policy(tmp_path)
        assert policy.authorize("$SIMUL_TEST_ROOT/scene.usd") == str(
            (tmp_path / "sandbox" / "scene.usd").resolve()
        )

    def test_dotdot_escape_is_refused(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        assert not policy.is_allowed(str(tmp_path / "sandbox" / ".." / "escape.usd"))

    def test_symlink_escape_is_refused(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        link = tmp_path / "sandbox" / "link"
        link.symlink_to(outside, target_is_directory=True)
        assert not policy.is_allowed(str(link / "escape.usd"))

    def test_empty_path_is_refused(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        assert not policy.is_allowed("")
        with pytest.raises(SandboxDenied):
            policy.authorize("")

    def test_disabled_policy_passes_everything_through_untouched(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path, enabled=False, allowed_url_schemes=())
        assert policy.authorize("relative/scene.usd") == "relative/scene.usd"
        assert policy.authorize("https://example.com/a.usd", write=True) == "https://example.com/a.usd"
        assert policy.is_allowed("")

    def test_denial_carries_message_and_details(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        with pytest.raises(SandboxDenied, match="sandbox") as excinfo:
            policy.authorize("/etc/shadow")
        details = excinfo.value.details
        assert details["file_path"] == "/etc/shadow"
        assert details["access"] == "read"
        assert str((tmp_path / "sandbox").resolve()) in details["allowed_roots"]
        assert details["allowed_url_schemes"] == ["omniverse"]
        assert "allowed_paths" in details["hint"]
        assert isinstance(excinfo.value, PermissionError)


class TestFromSettings:
    def test_settings_drive_the_scheme_allowlists(self) -> None:
        settings = Settings()
        security = settings.security.model_copy(
            update={
                "allowed_url_schemes": ["omniverse", "https"],
                "allowed_write_url_schemes": ["omniverse"],
            }
        )
        policy = PathPolicy.from_settings(settings.model_copy(update={"security": security}))
        assert policy.allowed_url_schemes == ("omniverse", "https")
        assert policy.allowed_write_url_schemes == ("omniverse",)
        assert policy.is_allowed("https://example.com/a.usd")
        assert policy.is_allowed("omniverse://nucleus/a.usd", write=True)
        assert not policy.is_allowed("https://example.com/a.usd", write=True)

    def test_default_settings_allow_omniverse_reads_only(self) -> None:
        policy = PathPolicy.from_settings(Settings())
        assert policy.allowed_url_schemes == ("omniverse",)
        assert policy.allowed_write_url_schemes == ()


class TestCaptureDirectory:
    def test_default_capture_dir_prefers_a_root_under_the_temp_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scratch output goes to scratch space even when it is not the first root."""
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        temp_root = temp_dir / "simul_mcp"
        policy = PathPolicy(
            enabled=True,
            allowed_paths=[str(tmp_path / "sandbox"), str(temp_root)],
            project_root=tmp_path,
        )
        assert policy.default_capture_dir() == str(temp_root.resolve() / "captures")
        assert policy.is_allowed(policy.default_capture_dir(), write=True)

    def test_default_capture_dir_falls_back_to_first_writable_root(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        assert policy.default_capture_dir() == str((tmp_path / "sandbox").resolve() / "captures")

    def test_default_capture_dir_is_none_when_no_root_is_writable(self, tmp_path: Path) -> None:
        unwritable = tmp_path / "locked"
        unwritable.mkdir()
        unwritable.chmod(0o500)
        try:
            if os.access(unwritable, os.W_OK):
                pytest.skip("filesystem ignores directory write bits (running as root?)")
            policy = PathPolicy(
                enabled=True,
                allowed_paths=[str(unwritable / "root")],
                project_root=tmp_path,
            )
            assert policy.writable_root() is None
            assert policy.default_capture_dir() is None
        finally:
            unwritable.chmod(0o700)

    def test_disabled_sandbox_uses_the_system_temp_dir(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path, enabled=False)
        expected = str(Path(tempfile.gettempdir()) / "simul_mcp" / "captures")
        assert policy.default_capture_dir() == expected
