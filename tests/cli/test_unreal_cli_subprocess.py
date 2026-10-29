"""Unreal CLI commands leave nothing for aiohttp to report at exit.

Every ``simul unreal`` command used to end its process with "Unclosed client
session" and "Unclosed connector" on stderr: the HTTP session was never
closed on the event loop that used it. aiohttp reports that only when the
process collects the session, so each command runs here as a fresh
subprocess, the way agents call it, against a local stand-in for Remote
Control.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from aiohttp import web

import simul

SRC = str(Path(simul.__file__).resolve().parents[1])
ACTOR = "/Game/Map.Map:PersistentLevel.PointLight_0"

COMMANDS: List[List[str]] = [
    ["health"],
    ["info"],
    ["exec", "print('hi')"],
    ["set-transform", ACTOR, "--location", "1,2,3"],
]


def _printed(text: str) -> Dict[str, Any]:
    """An ExecutePythonCommandEx reply for a script that printed ``text``."""
    return {
        "ReturnValue": True,
        "CommandResult": "None",
        "LogOutput": [{"Type": "Info", "Output": text + "\n"}],
    }


async def _remote_control(request: web.Request) -> web.Response:
    if request.method == "GET":
        return web.json_response({"HttpRoutes": []})
    code = (await request.json())["parameters"]["PythonCommand"]
    if "simul: set actor transform" in code:
        payload = {
            "actor_path": ACTOR,
            "location": [1.0, 2.0, 3.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
    elif "get_engine_version" in code:
        payload = {
            "engine_version": "5.7.4",
            "project_name": "Probe",
            "loaded_map": "/Temp/Untitled_1.Untitled_1",
            "platform": "Mac",
            "project_dir": "/tmp/Probe/",
        }
    else:
        return web.json_response(_printed("hi"))
    return web.json_response(_printed(json.dumps(payload)))


async def _run_commands() -> List[Tuple[List[str], int, str, str]]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", _remote_control)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = str(site._server.sockets[0].getsockname()[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
        "UNREAL__MODE": "endpoint",
        "SECURITY__ALLOW_SCRIPT_EXECUTION": "true",
    }

    async def run(argv: List[str]) -> Tuple[List[str], int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "simul.cli.main",
            "--json",
            "unreal",
            *argv,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await proc.communicate()
        return argv, proc.returncode or 0, out.decode(), err.decode()

    try:
        return await asyncio.gather(*(run(argv) for argv in COMMANDS))
    finally:
        await runner.cleanup()


def test_unreal_commands_exit_without_unclosed_session_warnings() -> None:
    for argv, code, out, err in asyncio.run(_run_commands()):
        assert code == 0, (argv, out, err)
        assert json.loads(out)["success"] is True, (argv, out)
        assert "Unclosed" not in err, (argv, err)
