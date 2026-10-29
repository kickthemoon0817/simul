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


def _build_server(backend: str, cwd: Path) -> Tuple[List[str], Set[str]]:
    """Build a server for ``backend`` alone in a fresh interpreter.

    Args:
        backend: The one backend name passed as the selection.
        cwd: Working directory, empty so no ``.env`` there feeds the settings.

    Returns:
        The backends whose tools registered, and every module then imported.
    """
    env = {
        key: value for key, value in os.environ.items() if key != "ISAAC_SIM__ENABLED"
    }
    overrides = json.dumps(SETTINGS_OVERRIDES.get(backend, {}))
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(SRC), backend, overrides],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
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


@pytest.mark.parametrize(
    "package_name", ["simul.mcp", "simul.mcp.schemas", "simul.utils"]
)
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
