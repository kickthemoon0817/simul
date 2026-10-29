"""A server imports only the backends it serves.

Building ``SimulMCPServer`` used to import every backend's code before the
``--backends`` selection applied (pxr for USD, aiohttp for Unreal, the Blender
adapter, numpy and every backend's schema models: about 0.25 s of the 0.6 s a
Blender, Unreal or Isaac server took to build), although only the selected
backends get an adapter and tools. The
backend registry now imports a backend's adapter and registration modules
when the server builds that backend, and ``simul.mcp``,
``simul.mcp.schemas`` and ``simul.utils`` resolve their re-exports on first
access. Each case builds a one-backend server in a fresh interpreter, so
``sys.modules`` shows exactly what construction imported.

Isaac Sim's adapter and tool object are built whatever the selection (they
own the instance registry's default client), so only its registration
modules count as Isaac's here.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# What only one backend needs: its runtime and its adapter, registration and
# schema modules. A server built for another backend must import none of them.
BACKEND_MODULES: Dict[str, Tuple[str, ...]] = {
    "usd": (
        "pxr",
        "numpy",
        "simul.usd",
        "simul.adapters.headless_usd",
        "simul.mcp.registration._reg_usd",
        "simul.mcp.schemas.usd",
    ),
    "blender": (
        "bpy",
        "simul.adapters.blender_runtime",
        "simul.adapters.blender_connection",
        "simul.blender_bridge",
        "simul.mcp.registration._reg_blender",
        "simul.mcp.schemas.blender",
        "simul.mcp.schemas.blender_ui",
        "simul.mcp.schemas.simready",
    ),
    "unreal": (
        "aiohttp",
        "PIL",
        "simul.adapters.unreal_runtime",
        "simul.adapters.unreal_connection",
        "simul.mcp.registration._reg_unreal",
        "simul.mcp.registration._unreal_tools",
        "simul.mcp.schemas.unreal",
        "simul.mcp.schemas.unreal_ui",
    ),
    "isaac": (
        "simul.mcp.registration._reg_isaac",
        "simul.mcp.registration._reg_instance",
    ),
}

# The packages whose re-exports resolve on first access (PEP 562).
LAZY_PACKAGES = ("simul.mcp", "simul.mcp.schemas", "simul.utils")

# Settings each backend needs to register its tools without an engine here:
# attached Blender needs no bpy in the server's interpreter.
SETTINGS_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "blender": {"blender": {"mode": "attached"}},
}

PROBE = textwrap.dedent("""
    import json, sys
    src, backend, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    sys.path.insert(0, src)
    from simul.config import Settings
    from simul.mcp.server import SimulMCPServer

    settings = Settings()
    for section, update in overrides.items():
        current = getattr(settings, section)
        settings = settings.model_copy(update={section: current.model_copy(update=update)})
    server = SimulMCPServer(settings, backends={backend})
    print(json.dumps({"registered": list(server._registered_backends), "modules": sorted(sys.modules)}))
    """)


def _run_fresh(probe: str, *args: str, cwd: Path) -> Dict[str, Any]:
    """Run ``probe`` in a fresh interpreter and return the JSON it printed last.

    Args:
        probe: Python source; it gets ``src`` as ``sys.argv[1]``, then ``args``.
        args: Further command-line arguments for the probe.
        cwd: Working directory, empty so no ``.env`` there feeds the settings.

    Returns:
        The object the probe's last line of output encodes.
    """
    env = {
        key: value for key, value in os.environ.items() if key != "ISAAC_SIM__ENABLED"
    }
    result = subprocess.run(
        [sys.executable, "-c", probe, str(SRC), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload: Dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return payload


def _build_server(backend: str, cwd: Path) -> Tuple[List[str], Set[str]]:
    """Build a server for ``backend`` alone in a fresh interpreter.

    Args:
        backend: The one backend name passed as the selection.
        cwd: Working directory, empty so no ``.env`` there feeds the settings.

    Returns:
        The backends whose tools registered, and every module then imported.
    """
    overrides = json.dumps(SETTINGS_OVERRIDES.get(backend, {}))
    payload = _run_fresh(PROBE, backend, overrides, cwd=cwd)
    return payload["registered"], set(payload["modules"])


@pytest.mark.parametrize("backend", sorted(BACKEND_MODULES))
def test_one_backend_server_imports_no_other_backend(
    backend: str, tmp_path: Path
) -> None:
    registered, loaded = _build_server(backend, tmp_path)

    # The backend registered its tools, so the check below cannot pass on a
    # server that built nothing.
    assert registered == [backend]
    foreign = sorted(
        module
        for other, modules in BACKEND_MODULES.items()
        if other != backend
        for module in modules
        if module in loaded
    )
    assert foreign == [], f"a {backend}-only server imported {foreign}"


@pytest.mark.parametrize("package_name", LAZY_PACKAGES)
def test_lazy_package_reexports_still_resolve(package_name: str) -> None:
    package = importlib.import_module(package_name)
    names = list(package.__all__)

    assert names and len(names) == len(set(names))
    for name in names:
        assert getattr(package, name) is not None, name
    assert set(names) <= set(dir(package))
    with pytest.raises(AttributeError):
        getattr(package, "NoSuchExport")


def test_schema_package_names_are_their_module_objects() -> None:
    from simul.mcp import schemas
    from simul.mcp.schemas import blender, common, instance, simready, unreal, usd

    modules = (common, usd, instance, simready, blender, unreal)
    assert schemas.__all__ == [name for module in modules for name in module.__all__]
    for module in modules:
        for name in module.__all__:
            assert getattr(schemas, name) is getattr(module, name), name

    namespace: Dict[str, Any] = {}
    exec("from simul.mcp.schemas import *", namespace)
    assert namespace["PrimInfo"] is usd.PrimInfo
    assert namespace["BlenderInfoResponse"] is blender.BlenderInfoResponse
    # A schema module is an attribute of the package even before anything
    # imported it, as when the package imported them all.
    for module_name in ("blender_ui", "unreal_ui"):
        module = importlib.import_module(f"simul.mcp.schemas.{module_name}")
        assert getattr(schemas, module_name) is module


def _type_checking_imports(package_name: str) -> Dict[str, str]:
    """Map each name a package imports under ``TYPE_CHECKING`` to its module.

    Args:
        package_name: A package whose re-exports resolve on first access.

    Returns:
        Name to the dotted name of the module it is imported from; a star
        import contributes every name in that module's ``__all__``.
    """
    package = importlib.import_module(package_name)
    tree = ast.parse(Path(str(package.__file__)).read_text(encoding="utf-8"))
    sources: Dict[str, str] = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for statement in node.body:
            assert isinstance(statement, ast.ImportFrom) and statement.level == 1
            module_name = f"{package_name}.{statement.module}"
            for alias in statement.names:
                if alias.name == "*":
                    star = importlib.import_module(module_name).__all__
                    sources.update(dict.fromkeys(star, module_name))
                else:
                    sources[alias.name] = module_name
    return sources


@pytest.mark.parametrize("package_name", LAZY_PACKAGES)
def test_type_checking_imports_name_every_lazy_export(package_name: str) -> None:
    # Type checkers read these imports instead of the runtime __getattr__, so
    # they must name exactly the package's exports, each from its own module.
    package = importlib.import_module(package_name)
    sources = _type_checking_imports(package_name)

    assert set(sources) == set(package.__all__)
    for name, module_name in sources.items():
        module = importlib.import_module(module_name)
        assert getattr(package, name) is getattr(module, name), name


SUBMODULE_PROBE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import simul.mcp
    import simul.utils

    print(json.dumps({
        "simul.mcp.schemas": simul.mcp.schemas.__name__,
        "simul.utils.math": simul.utils.math.__name__,
        "simul.utils.timing": simul.utils.timing.__name__,
    }))
    """)


def test_package_submodules_resolve_as_attributes(tmp_path: Path) -> None:
    # The package inits once imported these submodules, which left each one an
    # attribute of its package; they now resolve on first access instead.
    resolved = _run_fresh(SUBMODULE_PROBE, cwd=tmp_path)

    assert resolved == {
        "simul.mcp.schemas": "simul.mcp.schemas",
        "simul.utils.math": "simul.utils.math",
        "simul.utils.timing": "simul.utils.timing",
    }


PATCH_PROBE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from unittest import mock

    import pytest

    from simul import adapters
    from simul.config import Settings
    from simul.mcp import backends
    from simul.mcp.server import SimulMCPServer

    def registered():
        return list(SimulMCPServer(Settings(), backends={"usd"})._registered_backends)

    name = "is_headless_available"
    unresolved = name not in vars(backends)
    if sys.argv[2] == "monkeypatch":
        patcher = pytest.MonkeyPatch()
        patcher.setattr(backends, name, lambda: False)
        patched = registered()
        patcher.undo()
    else:
        with mock.patch(f"simul.mcp.backends.{name}", return_value=False):
            patched = registered()
    print(json.dumps({
        "unresolved": unresolved,
        "patched": patched,
        "restored": getattr(backends, name) is getattr(adapters, name),
        "real": registered(),
    }))
    """)


@pytest.mark.parametrize("patcher", ["monkeypatch", "mock"])
def test_patching_an_unresolved_probe_reaches_the_server(
    patcher: str, tmp_path: Path
) -> None:
    # The suite patches the probes on simul.mcp.backends. In a fresh
    # interpreter the patch lands before anything resolved the probe: the
    # server must still see it, and undoing it must restore the real probe.
    result = _run_fresh(PATCH_PROBE, patcher, cwd=tmp_path)

    assert result == {
        "unresolved": True,
        "patched": [],
        "restored": True,
        "real": ["usd"],
    }
