"""The bridge ext advertises whichever stock Python socket port this Isaac Sim ships."""

from __future__ import annotations


from khemoo.simul.mcp.extension import resolve_python_socket_port
from khemoo.simul.mcp.lifecycle import PYTHON_SERVER_TOKEN_HEADER


def test_prefers_python_server_setting_on_isaac_six() -> None:
    settings = {
        "/exts/isaacsim.code_editor.python_server/port": 8300,
        "/exts/isaacsim.code_editor.vscode/port": None,
    }
    assert resolve_python_socket_port(settings.get) == 8300


def test_falls_back_to_vscode_setting_on_isaac_five() -> None:
    settings = {"/exts/isaacsim.code_editor.vscode/port": 8226}
    assert resolve_python_socket_port(settings.get) == 8226


def test_defaults_when_neither_extension_is_configured() -> None:
    assert resolve_python_socket_port({}.get) == 8226


def test_token_header_constant_matches_python_server() -> None:
    assert PYTHON_SERVER_TOKEN_HEADER == "# isaacsim-python-server-token:"
