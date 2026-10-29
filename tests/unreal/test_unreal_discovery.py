"""Unreal discovery: one implementation behind ``simul unreal instances`` and ``list_unreal_instances``.

Fake Remote Control endpoints listen on a block of consecutive loopback ports,
served from a background event loop so the CLI's and the MCP tool's own
``asyncio.run`` calls can reach them. A port in the block without a fake is
released after binding, so connections to it are refused.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import random
import socket
import threading
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import aiohttp
import pytest
from aiohttp import web
from typer.testing import CliRunner

from simul.adapters import unreal_connection
from simul.adapters.unreal_connection import (
    UnrealAttachments,
    discover_instances,
    probe_instance,
)
from simul.adapters.unreal_runtime import (
    PASSPHRASE_REFUSED_ERROR,
    PROBE_NOT_SENT,
    UnrealRuntimeSession,
)
from simul.cli.main import app
from simul.config import Settings
from simul.mcp.schemas.unreal import UnrealListInstancesResponse
from simul.utils.private_files import BridgeFiles

try:
    import resource
except ImportError:  # Windows has no RLIMIT_NOFILE
    resource = None  # type: ignore[assignment]

VIEWPORTS = ["Layout.Viewport0", "Layout.Viewport1"]


def _identity(name: str, pid: int) -> Dict[str, Any]:
    """What the editor's identity script prints for a project called ``name``."""
    return {
        "instance_id": f"instance-{name}",
        "document_id": f"document-{name}",
        "pid": pid,
        "project_path": f"/projects/{name}/{name}.uproject",
        "map_path": "/Temp/Untitled_1.Untitled_1",
        "engine_version": "5.7.4",
        "viewports": list(VIEWPORTS),
        "active_viewport": VIEWPORTS[1],
        "dirty_map_packages": [],
        "project_name": name,
    }


class _FakeEditor:
    """One Remote Control endpoint: ``GET /remote/info`` plus Python execution."""

    def __init__(
        self,
        sock: socket.socket,
        identity: Optional[Dict[str, Any]] = None,
        python_status: int = 200,
        delay: float = 0.0,
        info_status: int = 200,
    ) -> None:
        self.sock = sock
        self.identity = identity
        self.python_status = python_status
        self.delay = delay
        # Non-200 refuses every route, as Remote Control does without the passphrase.
        self.info_status = info_status
        self.requests: List[Dict[str, Any]] = []

    def _record(self, request: web.Request) -> None:
        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
            }
        )

    def _refused(self) -> web.Response:
        return web.json_response(
            {"errorMessage": "Given Passphrase is not correct!"},
            status=self.info_status,
        )

    async def info(self, request: web.Request) -> web.Response:
        self._record(request)
        if self.info_status != 200:
            return self._refused()
        return web.json_response({"HttpRoutes": []})

    async def call(self, request: web.Request) -> web.Response:
        self._record(request)
        if self.info_status != 200:
            return self._refused()
        body = await request.json()
        assert body["functionName"] == "ExecutePythonCommandEx"
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.python_status != 200:
            # Remote Control refusing the Python library, e.g. remote Python disabled.
            return web.json_response(
                {"errorMessage": "Python disabled"}, status=self.python_status
            )
        if self.identity is None:
            return web.json_response(
                {
                    "ReturnValue": False,
                    "CommandResult": "RuntimeError: No editor world is available",
                }
            )
        return web.json_response(
            {
                "ReturnValue": True,
                "CommandResult": "None",
                "LogOutput": [
                    {"Type": "Info", "Output": json.dumps(self.identity) + "\n"}
                ],
            }
        )


def _bind_block(count: int) -> List[socket.socket]:
    """Bind ``count`` consecutive loopback ports so one scan range covers them."""
    for _ in range(100):
        base = random.randint(20000, 60000)
        socks: List[socket.socket] = []
        try:
            for offset in range(count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                socks.append(sock)
                sock.bind(("127.0.0.1", base + offset))
        except OSError:
            for sock in socks:
                sock.close()
            continue
        return socks
    raise RuntimeError("No block of free loopback ports")


class _Editors:
    """Fake editors on consecutive ports; ``None`` in ``specs`` is a refused port."""

    def __init__(self, specs: List[Optional[Dict[str, Any]]]) -> None:
        self.socks = _bind_block(len(specs))
        self.ports = [sock.getsockname()[1] for sock in self.socks]
        self.by_port = {
            port: _FakeEditor(sock, **spec)
            for port, sock, spec in zip(self.ports, self.socks, specs)
            if spec is not None
        }
        # A bound socket that never listens drops connections (a timeout on
        # macOS) instead of refusing them, so free the refused ports.
        for sock, spec in zip(self.socks, specs):
            if spec is None:
                sock.close()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.runners: List[web.AppRunner] = []

    async def _start(self) -> None:
        for editor in self.by_port.values():
            application = web.Application()
            application.router.add_get("/remote/info", editor.info)
            application.router.add_put("/remote/object/call", editor.call)
            runner = web.AppRunner(application)
            await runner.setup()
            await web.SockSite(runner, editor.sock).start()
            self.runners.append(runner)

    async def _stop(self) -> None:
        for runner in self.runners:
            await runner.cleanup()

    def __enter__(self) -> "_Editors":
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(10)
        return self

    def __exit__(self, *exc: Any) -> None:
        asyncio.run_coroutine_threadsafe(self._stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        self.loop.close()
        for sock in self.socks:
            sock.close()

    def settings(self, tmp_path: Any, active: int, **unreal: Any) -> Settings:
        """Settings scanning exactly this block, with ``active`` the configured port."""
        base = Settings()
        update = {
            "mode": "endpoint",
            "host": "127.0.0.1",
            "port": self.ports[active],
            "scan_port_start": self.ports[0],
            "scan_port_end": self.ports[-1] + 1,
            "attachment_path": str(tmp_path / "attachment.json"),
            "ping_timeout": 2.0,
            "passphrase": None,
            **unreal,
        }
        return base.model_copy(update={"unreal": base.unreal.model_copy(update=update)})

    def attach_record(self, tmp_path: Any, index: int, **target: Any) -> None:
        """Save an attachment to the editor at ``index`` (``target`` overrides identity)."""
        identity = self.by_port[self.ports[index]].identity
        assert identity is not None
        BridgeFiles.write(
            tmp_path / "attachment.json",
            {
                "version": 1,
                "host": "127.0.0.1",
                "port": self.ports[index],
                "target": {
                    **{
                        k: identity[k]
                        for k in (
                            "instance_id",
                            "document_id",
                            "project_path",
                            "map_path",
                        )
                    },
                    "viewport": VIEWPORTS[0],
                    **target,
                },
            },
        )


# Refused (configured port), editor "alpha", refused, Remote Control without Python, editor "beta".
_MIXED = [
    None,
    {"identity": _identity("alpha", 101)},
    None,
    {"python_status": 400},
    {"identity": _identity("beta", 202)},
]


@pytest.fixture
def mixed() -> Iterator[_Editors]:
    with _Editors(_MIXED) as editors:
        yield editors


def _without_latency(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The payload minus per-run round-trip times, which differ between two calls."""
    return {
        **payload,
        "instances": [
            {k: v for k, v in item.items() if k != "latency_ms"}
            for item in payload["instances"]
        ],
    }


def _cli(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, *args: str
) -> Dict[str, Any]:
    monkeypatch.setattr("simul.cli.unreal_cli.get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["--json", "unreal", "instances", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _tool(settings: Settings, name: str, **arguments: Any) -> Dict[str, Any]:
    """Call an MCP tool registered on a server built over ``settings``."""
    from simul.mcp.server import SimulMCPServer

    server = SimulMCPServer(settings, backends={"unreal"})
    response = asyncio.run(server.mcp.by_name[name](**arguments))
    return json.loads(response.content[-1].text)


def _mcp(settings: Settings, **arguments: Any) -> Dict[str, Any]:
    return _tool(settings, "list_unreal_instances", **arguments)


def test_scan_reports_identity_and_flags_and_skips_silent_ports(
    mixed: _Editors, tmp_path: Any
) -> None:
    mixed.attach_record(tmp_path, 4)
    payload = asyncio.run(discover_instances(mixed.settings(tmp_path, active=0)))
    p = mixed.ports
    assert [item["port"] for item in payload["instances"]] == [p[0], p[1], p[3], p[4]]
    configured, alpha, no_python, beta = payload["instances"]

    # The configured endpoint is listed even though nothing answers there.
    assert configured["name"] == "default" and configured["active"] is True
    assert configured["reachable"] is False and configured["error"]
    assert configured["instance_id"] is None and configured["viewports"] == []

    assert alpha["name"] == f"unreal-{p[1]}"
    assert (alpha["active"], alpha["attached"], alpha["reachable"]) == (
        False,
        False,
        True,
    )
    assert isinstance(alpha["latency_ms"], float) and alpha["error"] is None
    assert {k: alpha[k] for k in _identity("alpha", 101)} == _identity("alpha", 101)

    # Remote Control answers but Python execution is refused: reachable, no identity, a reason.
    assert no_python["reachable"] is True and no_python["instance_id"] is None
    assert "Editor identity unavailable" in no_python["error"]
    assert "simul unreal setup" in no_python["error"]

    # Endpoint mode still reports which editor the saved attachment selects.
    assert beta["attached"] is True and beta["active"] is False

    assert payload["active_instance"] == "default"
    assert payload["total_discovered"] == 3
    assert payload["warnings"] == [] and payload["success"] is True


def test_payload_is_exactly_the_mcp_schema(mixed: _Editors, tmp_path: Any) -> None:
    """The MCP envelope validates the payload through the schema; nothing may be added or dropped."""
    payload = asyncio.run(discover_instances(mixed.settings(tmp_path, active=0)))
    assert UnrealListInstancesResponse(**payload).model_dump() == payload


@pytest.mark.parametrize(
    "cli_args,tool_args,listed",
    [((), {}, [1, 3, 4]), (("--no-scan",), {"scan": False}, [1])],
)
def test_cli_and_mcp_return_the_same_payload(
    mixed: _Editors,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
    cli_args: tuple,
    tool_args: Dict[str, Any],
    listed: List[int],
) -> None:
    settings = mixed.settings(tmp_path, active=1)
    mixed.attach_record(tmp_path, 4)
    from_cli = _cli(monkeypatch, settings, *cli_args)
    from_mcp = _mcp(settings, **tool_args)
    assert _without_latency(from_cli) == _without_latency(from_mcp)
    assert [item["port"] for item in from_mcp["instances"]] == [
        mixed.ports[i] for i in listed
    ]
    first = from_mcp["instances"][0]
    assert (first["name"], first["active"], first["project_name"]) == (
        "default",
        True,
        "alpha",
    )
    assert [item["attached"] for item in from_mcp["instances"]] == [
        i == 4 for i in listed
    ]


def test_no_scan_probes_only_the_active_endpoint(
    mixed: _Editors, tmp_path: Any
) -> None:
    payload = asyncio.run(
        discover_instances(mixed.settings(tmp_path, active=1), scan=False)
    )
    assert [item["port"] for item in payload["instances"]] == [mixed.ports[1]]
    assert payload["instances"][0]["pid"] == 101
    assert mixed.by_port[mixed.ports[4]].requests == []


def test_explicit_port_reports_its_endpoint_even_when_refused(
    mixed: _Editors, tmp_path: Any
) -> None:
    settings = mixed.settings(tmp_path, active=1)
    payload = asyncio.run(discover_instances(settings, port=mixed.ports[2]))
    (only,) = payload["instances"]
    assert (
        only["port"] == mixed.ports[2] and only["reachable"] is False and only["error"]
    )
    assert payload["total_discovered"] == 0 and payload["active_instance"] is None


def test_attached_mode_targets_the_attachment(mixed: _Editors, tmp_path: Any) -> None:
    settings = mixed.settings(tmp_path, active=0, mode="attached")
    mixed.attach_record(tmp_path, 4)
    payload = asyncio.run(discover_instances(settings))
    by_port = {item["port"]: item for item in payload["instances"]}
    # The configured port is not the target in attached mode, so its silence is not reported.
    assert mixed.ports[0] not in by_port
    beta = by_port[mixed.ports[4]]
    assert (beta["name"], beta["active"], beta["attached"]) == ("default", True, True)
    assert payload["warnings"] == []

    # Same editor, new map generation: still the dispatch target, no longer the attachment.
    mixed.attach_record(tmp_path, 4, document_id="an-earlier-map")
    payload = asyncio.run(discover_instances(settings))
    beta = next(item for item in payload["instances"] if item["active"])
    assert beta["attached"] is False
    assert "simul unreal attach" in payload["warnings"][0]

    # No attachment: nothing is active, discovery still lists the editors and says why.
    (tmp_path / "attachment.json").unlink()
    payload = asyncio.run(discover_instances(settings))
    assert payload["active_instance"] is None
    assert [
        item["project_name"] for item in payload["instances"] if item["instance_id"]
    ] == ["alpha", "beta"]
    assert "No Unreal editor attached" in payload["warnings"][0]
    assert asyncio.run(discover_instances(settings, scan=False))["instances"] == []


@pytest.mark.parametrize(
    "spec,reason",
    [
        # Busy past the timeout (an editor still starting up).
        (
            {"identity": _identity("slow", 7), "delay": 1.0},
            "did not answer within 0.3s",
        ),
        # Remote Control refuses the Python library.
        ({"python_status": 400}, "simul unreal setup"),
        # The identity script raised inside the editor.
        ({}, "No editor world is available"),
    ],
)
def test_an_editor_without_identity_stays_reachable_and_says_why(
    tmp_path: Any, spec: Dict[str, Any], reason: str
) -> None:
    with _Editors([spec]) as editors:
        settings = editors.settings(tmp_path, active=0, ping_timeout=0.3)
        record = asyncio.run(probe_instance(settings, "127.0.0.1", editors.ports[0]))
    assert record["reachable"] is True and isinstance(record["latency_ms"], float)
    assert record["instance_id"] is None and record["viewports"] == []
    assert reason in record["error"]


@pytest.mark.parametrize(
    "passphrase,header",
    # UE compares the header with the MD5 of the plaintext (FMD5::HashAnsiString).
    [(None, None), ("password", "5f4dcc3b5aa765d61d8327deb882cf99")],
)
def test_every_probe_carries_the_configured_passphrase(
    tmp_path: Any, passphrase: Optional[str], header: Optional[str]
) -> None:
    """A passphrase-enforcing editor rejects requests without the header, so discovery must send it."""
    with _Editors([{"identity": _identity("auth", 9)}]) as editors:
        settings = editors.settings(tmp_path, active=0, passphrase=passphrase)
        payload = asyncio.run(discover_instances(settings))
        requests = editors.by_port[editors.ports[0]].requests
    assert payload["instances"][0]["pid"] == 9
    assert [request["path"] for request in requests] == [
        "/remote/info",
        "/remote/object/call",
    ]
    assert {request["headers"].get("Passphrase") for request in requests} == {header}


@pytest.mark.parametrize("status,listed", [(401, True), (404, False)])
def test_a_scan_lists_an_editor_that_refuses_the_passphrase(
    tmp_path: Any, status: int, listed: bool
) -> None:
    """Remote Control answers 401 to a missing or wrong passphrase: an editor is
    there, so a scan lists it with the reason instead of dropping it. Other
    refusals need not come from Unreal and stay out of the scan."""
    with _Editors([None, {"info_status": status}]) as editors:
        settings = editors.settings(tmp_path, active=0)
        payload = asyncio.run(discover_instances(settings))
        requests = editors.by_port[editors.ports[1]].requests
        if listed:
            with pytest.raises(ValueError, match="UNREAL__PASSPHRASE"):
                asyncio.run(UnrealAttachments(settings).attach())
    # The refusal ends the probe; no identity script is sent.
    assert {request["path"] for request in requests} == {"/remote/info"}
    found = [item for item in payload["instances"] if item["port"] == editors.ports[1]]
    assert payload["total_discovered"] == int(listed)
    if not listed:
        assert found == []
        return
    (refused,) = found
    assert (refused["reachable"], refused["active"]) == (True, False)
    assert refused["instance_id"] is None and refused["viewports"] == []
    assert "HTTP 401" in refused["error"] and "UNREAL__PASSPHRASE" in refused["error"]


def test_attach_explains_an_editor_without_identity(tmp_path: Any) -> None:
    with _Editors([{"python_status": 400}]) as editors:
        manager = UnrealAttachments(editors.settings(tmp_path, active=0))
        with pytest.raises(ValueError, match="Reachable without identity"):
            asyncio.run(manager.attach())
        with pytest.raises(RuntimeError, match="simul unreal setup"):
            asyncio.run(manager.describe("127.0.0.1", editors.ports[0]))
    assert not manager.path.exists()


def test_attach_selects_from_discovery_and_rechecks(
    mixed: _Editors, tmp_path: Any
) -> None:
    manager = UnrealAttachments(mixed.settings(tmp_path, active=0))
    selected = asyncio.run(manager.attach(port=mixed.ports[1], viewport=VIEWPORTS[1]))
    assert (
        selected["instance_id"] == "instance-alpha"
        and selected["viewport"] == VIEWPORTS[1]
    )
    assert "error" not in selected and selected["mode"] == "attached"
    with pytest.raises(ValueError, match="Unknown viewport"):
        asyncio.run(manager.attach(port=mixed.ports[1], viewport="Layout.Missing"))
    with pytest.raises(ValueError, match="unique Unreal"):
        asyncio.run(manager.attach())  # alpha and beta both answer
    payload = asyncio.run(discover_instances(mixed.settings(tmp_path, active=0)))
    assert [item["attached"] for item in payload["instances"]] == [
        False,
        True,
        False,
        False,
    ]


# ---------------------------------------------------------------------------
# One definition of reachable: ping_unreal, unreal_health_check and discovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,reachable",
    [
        ({"identity": _identity("ok", 5)}, True),
        ({"info_status": 401}, True),
        (None, False),
    ],
    ids=["answers", "refuses-the-passphrase", "nothing-listening"],
)
def test_ping_health_and_discovery_agree_on_reachable(
    tmp_path: Any,
    fake_fastmcp: Any,
    spec: Optional[Dict[str, Any]],
    reachable: bool,
) -> None:
    with _Editors([spec]) as editors:
        settings = editors.settings(tmp_path, active=0, max_retries=0)
        ping = _tool(settings, "ping_unreal")
        health = _tool(settings, "unreal_health_check")
        (instance,) = _tool(settings, "list_unreal_instances", scan=False)["instances"]
    assert (ping["reachable"], health["reachable"], instance["reachable"]) == (
        reachable,
        reachable,
        reachable,
    )


def test_a_refusal_is_reachable_with_the_same_authentication_error_everywhere(
    tmp_path: Any, fake_fastmcp: Any
) -> None:
    """Remote Control answered 401: reachable, and every tool names the passphrase."""
    with _Editors([{"info_status": 401}]) as editors:
        settings = editors.settings(tmp_path, active=0)
        ping = _tool(settings, "ping_unreal")
        health = _tool(settings, "unreal_health_check")
        (instance,) = _tool(settings, "list_unreal_instances", scan=False)["instances"]
    for payload in (ping, health):
        assert payload["success"] is False and payload["reachable"] is True
        assert (payload["error_type"], payload["status"]) == (
            "AuthenticationError",
            401,
        )
    assert (
        ping["error"]
        == health["error"]
        == instance["error"]
        == PASSPHRASE_REFUSED_ERROR
    )
    assert isinstance(ping["latency_ms"], float) and isinstance(
        instance["latency_ms"], float
    )
    # The deprecated alias keeps meaning "usable" for callers still gating on it.
    assert health["connected"] is False


# ---------------------------------------------------------------------------
# A scan holds a bounded number of sockets and reports probes it never sent
# ---------------------------------------------------------------------------


def _scan_settings(tmp_path: Any, start: int, end: int, **unreal: Any) -> Settings:
    base = Settings()
    update = {
        "mode": "endpoint",
        "host": "127.0.0.1",
        "port": start,
        "scan_port_start": start,
        "scan_port_end": end,
        "attachment_path": str(tmp_path / "attachment.json"),
        "passphrase": None,
        **unreal,
    }
    return base.model_copy(update={"unreal": base.unreal.model_copy(update=update)})


def test_a_scan_has_at_most_max_concurrent_probes_in_flight(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_flight = peak = 0

    async def probe(settings: Settings, host: str, port: int) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.005)
        in_flight -= 1
        record = {
            "host": host,
            "port": port,
            "reachable": False,
            "latency_ms": None,
            "error": "refused",
            **{key: None for key in unreal_connection._IDENTITY_KEYS},
            **{key: [] for key in unreal_connection._IDENTITY_LISTS},
        }
        return record, {"reachable": False, "error": "refused"}

    monkeypatch.setattr(unreal_connection, "_probe", probe)
    payload = asyncio.run(discover_instances(_scan_settings(tmp_path, 40000, 40500)))

    assert peak == unreal_connection.MAX_CONCURRENT_PROBES
    assert [item["port"] for item in payload["instances"]] == [40000]


@pytest.mark.skipif(
    resource is None or not os.path.isdir("/dev/fd"),
    reason="needs RLIMIT_NOFILE and /dev/fd",
)
def test_a_wide_scan_under_a_low_descriptor_limit_still_finds_the_editor(
    tmp_path: Any,
) -> None:
    """Probing every port at once ran out of descriptors (macOS allows 256 by
    default), and the failed connects hid an editor late in the range."""
    with _Editors([None] * 299 + [{"identity": _identity("late", 3)}]) as editors:
        settings = editors.settings(tmp_path, active=0)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Room for the bounded probes, far fewer descriptors than ports.
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (len(os.listdir("/dev/fd")) + 64, hard)
        )
        try:
            payload = asyncio.run(discover_instances(settings))
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    found = {item["port"]: item for item in payload["instances"]}
    assert found[editors.ports[-1]]["project_name"] == "late"
    assert payload["warnings"] == []


_KEY = SimpleNamespace(host="127.0.0.1", port=30010, ssl=True)
_DNS_ERROR = getattr(aiohttp, "ClientConnectorDNSError", aiohttp.ClientConnectorError)


@pytest.mark.parametrize(
    "exc,not_sent",
    [
        (
            aiohttp.ClientConnectorError(
                _KEY, OSError(errno.EMFILE, "Too many open files")
            ),
            True,
        ),
        (
            aiohttp.ClientConnectorError(
                _KEY, OSError(errno.EADDRNOTAVAIL, "No local port")
            ),
            True,
        ),
        (
            _DNS_ERROR(
                _KEY, socket.gaierror(socket.EAI_NONAME, "Name does not resolve")
            ),
            True,
        ),
        (OSError(errno.ENFILE, "Too many open files in system"), True),
        (
            aiohttp.ClientConnectorError(
                _KEY, ConnectionRefusedError(errno.ECONNREFUSED, "Refused")
            ),
            False,
        ),
        (asyncio.TimeoutError(), False),
    ],
    ids=["emfile", "no-local-port", "dns", "enfile", "refused", "timeout"],
)
def test_ping_says_when_its_request_never_left_this_machine(
    exc: BaseException, not_sent: bool
) -> None:
    session = UnrealRuntimeSession(Settings())

    async def http_get(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise exc

    session._http_get = http_get  # type: ignore[method-assign]
    result = asyncio.run(session.ping())

    assert result["reachable"] is False and result["error"]
    assert (result.get("error_type") == PROBE_NOT_SENT) is not_sent


def test_a_scan_names_the_ports_it_could_not_probe(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that never left this machine says nothing about its port, so the
    scan reports the port instead of dropping it like a silent one."""
    start = 40000
    unsent = {start + 1, start + 2, start + 4}

    async def http_get(
        self: UnrealRuntimeSession, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        key = SimpleNamespace(host=self.host, port=self.port, ssl=True)
        if self.port in unsent:
            raise aiohttp.ClientConnectorError(
                key, OSError(errno.EMFILE, "Too many open files")
            )
        raise aiohttp.ClientConnectorError(
            key, ConnectionRefusedError(errno.ECONNREFUSED, "Refused")
        )

    monkeypatch.setattr(UnrealRuntimeSession, "_http_get", http_get)
    payload = asyncio.run(
        discover_instances(_scan_settings(tmp_path, start, start + 6))
    )

    # The active endpoint is listed as ever; silent ports stay out.
    assert [item["port"] for item in payload["instances"]] == [start]
    (warning,) = payload["warnings"]
    assert warning.startswith("Probe not sent ([Errno 24] Too many open files)")
    assert f"127.0.0.1 port(s) {start + 1}-{start + 2}, {start + 4}" in warning
    assert "missing from instances" in warning


@pytest.fixture
def runtime_log() -> Iterator[List[logging.LogRecord]]:
    """Every record UnrealRuntimeSession logs.

    Tests elsewhere call ``setup_logging``, whose dictConfig can disable this
    logger and stop propagation to caplog's handler, so listen on it directly.
    """
    records: List[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("simul.adapters.unreal_runtime.UnrealRuntimeSession")
    handler = _Handler(level=logging.DEBUG)
    saved = (logger.disabled, logger.level)
    logger.disabled = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.disabled = saved[0]
        logger.setLevel(saved[1])


def test_an_editor_without_identity_is_logged_at_debug_with_the_reason_kept(
    tmp_path: Any, runtime_log: List[logging.LogRecord]
) -> None:
    """Normal while an editor starts up: no error in the server log, and the
    reason stays in the result."""
    with _Editors([{}]) as editors:  # the identity script raises: no editor world yet
        settings = editors.settings(tmp_path, active=0)
        record = asyncio.run(probe_instance(settings, "127.0.0.1", editors.ports[0]))

    assert record["reachable"] is True and record["instance_id"] is None
    assert "No editor world is available" in record["error"]
    logged = [r.levelno for r in runtime_log if "No editor world" in r.getMessage()]
    assert logged == [logging.DEBUG]


def test_other_tool_script_failures_are_still_logged_as_errors(
    runtime_log: List[logging.LogRecord],
) -> None:
    session = UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return {"ReturnValue": False, "CommandResult": "RuntimeError: boom"}

    session._execute_python = execute  # type: ignore[method-assign]
    result = asyncio.run(session._execute_json_script("raise RuntimeError('boom')"))

    assert result["error_type"] == "ScriptError"
    assert [r.levelno for r in runtime_log if "boom" in r.getMessage()] == [
        logging.ERROR
    ]
