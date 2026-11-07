"""Remote Control failures reach the caller with Unreal's reason, in one taxonomy.

The fake editors here are real HTTP servers that answer as UE 5.7's Remote
Control does: an error body built by an unescaped printf
(``{ "errorMessage": "%s" }``, so a multi-line message is not valid JSON), 400
for a name that resolves to nothing on most routes and 404 on others, and a
batch route that dies without a Passphrase header
(FWebRemoteControlModule::HandleBatchRequest reads it unchecked).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
from aiohttp import web

from simul.adapters import unreal_runtime
from simul.adapters._unreal_errors import (
    UnrealConnectionError,
    UnrealError,
    UnrealTimeoutError,
    remote_control_error_type,
    remote_control_message,
    script_not_found,
)
from simul.adapters.unreal_runtime import (
    BATCH_PASSPHRASE_PLACEHOLDER,
    MAX_BATCH_OPERATIONS,
    SET_CONSOLE_VARIABLE,
    UnrealRuntimeSession,
)
from simul.config import Settings

CUBE = "/Game/Maps/Main.Main:PersistentLevel.StaticMeshActor_1"

#: A Remote Control property write: a mutation, so it is sent once.
WRITE_TAGS = {"objectPath": CUBE, "propertyName": "Tags", "propertyValue": {"Tags": ["a"]},
              "access": "WRITE_TRANSACTION_ACCESS"}


def _settings(port: int, **unreal: Any) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "unreal": base.unreal.model_copy(
                update={"host": "127.0.0.1", "port": port, "timeout": 2.0, "retry_base_delay": 0.001, **unreal}
            ),
            "security": base.security.model_copy(update={"allow_script_execution": True}),
        }
    )


@asynccontextmanager
async def editor(handler: Any, **unreal: Any) -> AsyncIterator[UnrealRuntimeSession]:
    """A session on a fake Remote Control endpoint that ``handler`` answers."""

    async def route(request: web.Request) -> web.StreamResponse:
        return await handler(request)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", route)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    session = UnrealRuntimeSession(_settings(site._server.sockets[0].getsockname()[1], **unreal))
    try:
        yield session
    finally:
        await session.close()
        await runner.cleanup()


def rc_error(message: str, status: int) -> web.Response:
    """An error answer built as UE 5.7 builds it: printf'd, newlines and all."""
    return web.Response(
        text='{ "errorMessage": "%s" }' % message, status=status, content_type="application/json"
    )


def python_reply(printed: Dict[str, Any]) -> web.Response:
    return web.json_response({"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": json.dumps(printed)}]})


# -- U1-01: a batch always carries a Passphrase header ------------------------


class _BatchEditor:
    """Answers a batch only when it has a Passphrase header, as UE 5.7 needs."""

    def __init__(self) -> None:
        self.batches: List[Dict[str, Any]] = []
        self.headers: List[Optional[str]] = []
        self.crashed = False

    async def __call__(self, request: web.Request) -> web.StreamResponse:
        if request.path != "/remote/batch":
            self.headers.append(request.headers.get("Passphrase"))
            return web.json_response({})
        passphrase = request.headers.get("Passphrase", "")
        if not passphrase:
            # UE drops a header with an empty value, then asserts on the
            # missing key and aborts.
            self.crashed = True
            raise ConnectionResetError("Assertion failed: Pair != nullptr")
        body = await request.json()
        self.batches.append(body)
        self.headers.append(passphrase)
        return web.json_response(
            {
                "Responses": [
                    {"RequestId": op["RequestId"], "ResponseCode": 200, "ResponseBody": {"ReturnValue": "5.7.4"}}
                    for op in body["Requests"]
                ]
            }
        )


ENGINE_VERSION_OP = {
    "RequestId": 1,
    "URL": "/remote/object/call",
    "Verb": "PUT",
    "Body": {"objectPath": "/Script/Engine.Default__KismetSystemLibrary", "functionName": "GetEngineVersion"},
}


@pytest.mark.asyncio
async def test_a_batch_without_a_configured_passphrase_carries_the_placeholder() -> None:
    fake = _BatchEditor()
    async with editor(fake) as session:
        result = await session.batch_operations([ENGINE_VERSION_OP])
        await session._http_get("/remote/info")

    assert fake.crashed is False
    assert result["succeeded"] == 1 and result["failed"] == 0
    # Only the batch carries it: every other route checks for the header.
    assert fake.headers == [BATCH_PASSPHRASE_PLACEHOLDER, None]


@pytest.mark.asyncio
async def test_a_batch_with_a_configured_passphrase_carries_its_hash() -> None:
    fake = _BatchEditor()
    async with editor(fake, passphrase="password") as session:
        await session.batch_operations([ENGINE_VERSION_OP])
        await session._http_get("/remote/info")

    digest = hashlib.md5(b"password").hexdigest()
    assert fake.headers == [digest, digest]


def test_the_placeholder_is_not_empty_and_never_matches_a_passphrase() -> None:
    # Empty would be dropped by UE's HTTP server; an MD5 digest could match.
    assert BATCH_PASSPHRASE_PLACEHOLDER.strip()
    assert unreal_runtime._passphrase_to_md5(BATCH_PASSPHRASE_PLACEHOLDER) != BATCH_PASSPHRASE_PLACEHOLDER


@pytest.mark.asyncio
async def test_batch_operations_are_sent_in_the_order_remote_control_reads_them() -> None:
    fake = _BatchEditor()
    operations = [
        # Body before RequestId: UE would hand this Body to the previous id.
        {"body": {"objectPath": CUBE, "functionName": "K2_GetActorLocation"}, "url": "/remote/object/call",
         "verb": "put", "requestid": 7},
        {"URL": "/remote/info", "Verb": "GET"},
        {"URL": "/remote/object/describe", "Verb": "PUT", "Body": {"objectPath": CUBE}},
    ]
    async with editor(fake) as session:
        result = await session.batch_operations(json.dumps(operations))

    (sent,) = fake.batches
    assert [list(op) for op in sent["Requests"]] == [
        ["RequestId", "URL", "Verb", "Body"],
        ["RequestId", "URL", "Verb"],
        ["RequestId", "URL", "Verb", "Body"],
    ]
    # Operations without an id take the free ones, in order; verbs are upper case.
    assert [(op["RequestId"], op["Verb"]) for op in sent["Requests"]] == [(7, "PUT"), (1, "GET"), (2, "PUT")]
    assert result["total"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operations,reason",
    [
        ("not json", "not valid JSON"),
        ("{}", "non-empty JSON array"),
        ([], "non-empty JSON array"),
        (["/remote/info"], "must be an object"),
        ([{"URL": "/remote/info"}], "Verb must be one of"),
        ([{"URL": "/remote/info", "Verb": "TRACE"}], "Verb must be one of"),
        ([{"URL": "remote/info", "Verb": "GET"}], "needs URL"),
        ([{"URL": "/other/route", "Verb": "GET"}], "needs URL"),
        ([{"URL": "/remote/Batch", "Verb": "PUT", "Body": {}}], "cannot contain another"),
        ([{"URL": "/remote/info", "Verb": "GET", "Body": "text"}], "Body must be a JSON object"),
        ([{"URL": "/remote/info", "Verb": "GET", "RequestId": "1"}], "RequestId must be an integer"),
        ([{"URL": "/remote/info", "Verb": "GET", "RequestId": True}], "RequestId must be an integer"),
        ([{"URL": "/remote/info", "Verb": "GET", "Url": "/remote/info"}], "gives URL twice"),
        ([{"URL": "/remote/info", "Verb": "GET", "Payload": {}}], "unknown key 'Payload'"),
        ([{"URL": "/remote/info", "Verb": "GET", "RequestId": 1}] * 2, "must be unique"),
        ([{"URL": "/remote/info", "Verb": "GET"}] * (MAX_BATCH_OPERATIONS + 1), "at most"),
    ],
)
async def test_a_malformed_batch_is_refused_before_anything_is_sent(operations: Any, reason: str) -> None:
    fake = _BatchEditor()
    async with editor(fake) as session:
        result = await session.batch_operations(operations)

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert reason in result["error"]
    assert fake.batches == [] and fake.headers == []


@pytest.mark.asyncio
async def test_a_failed_batch_operation_says_why() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "Responses": [
                    {"RequestId": 1, "ResponseCode": 200, "ResponseBody": {"ReturnValue": {"X": 0}}},
                    {"RequestId": 2, "ResponseCode": 400,
                     "ResponseBody": {"errorMessage": f"Object: {CUBE}_nope does not exist."}},
                    {"RequestId": 3, "ResponseCode": 400,
                     "ResponseBody": {"errorMessage": "Property is not writeable"}},
                ]
            }
        )

    op = {"URL": "/remote/object/call", "Verb": "PUT", "Body": {}}
    async with editor(handler) as session:
        result = await session.batch_operations([op, op, op])

    assert (result["succeeded"], result["failed"]) == (1, 2)
    assert "error" not in result["results"][0]
    assert result["results"][1]["error_type"] == "NotFoundError"
    assert result["results"][1]["error"] == f"Object: {CUBE}_nope does not exist."
    assert result["results"][2]["error_type"] == "ValidationError"


# -- U1-07: Remote Control's reason, in the shared taxonomy --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,message,error_type",
    [
        (400, f"Object property: NoSuchProperty could not be resolved on object: {CUBE}", "NotFoundError"),
        (400, f"Object: {CUBE}_nope does not exist when trying to resolve property: Tags", "NotFoundError"),
        (400, f"Function: NoSuchFunction does not exist on object: {CUBE}", "NotFoundError"),
        (404, "Unable to find the object.", "NotFoundError"),
        (
            400,
            "Property is not writeable because it is protected without the AllowPrivateAccess specifier set to "
            "true or a setter assigned.\nProperty is not writeable because it has the EditConst flag.",
            "ValidationError",
        ),
        (400, "Unable to deserialize request.", "ValidationError"),
        (400, f"Can't resolve object: {CUBE} while saving or garbage collecting.", "InstanceBusy"),
        (401, "Given Passphrase is not correct!", "AuthenticationError"),
        (500, "", "EngineError"),
    ],
)
async def test_remote_control_refusals_keep_their_message(status: int, message: str, error_type: str) -> None:
    async def handler(request: web.Request) -> web.Response:
        return rc_error(message, status)

    async with editor(handler, max_retries=0) as session:
        with pytest.raises(UnrealError) as raised:
            await session._http_put("/remote/object/property", WRITE_TAGS)

    error = raised.value
    assert error.error_type == error_type
    assert error.status == status
    assert str(error).startswith(f"Remote Control refused PUT /remote/object/property (HTTP {status})")
    assert message in str(error)
    assert error.details == {
        "status": status,
        "route": "/remote/object/property",
        "remote_control_message": message or None,
    }


def test_the_message_is_read_from_valid_and_printf_built_bodies() -> None:
    assert remote_control_message('{"errorMessage": "plain"}') == "plain"
    assert remote_control_message('{ "errorMessage": "two\nlines" }') == "two\nlines"
    assert remote_control_message('{ "errorMessage": "say "hi"" }') == 'say "hi"'
    assert remote_control_message("not json at all") == "not json at all"
    assert remote_control_message("") == ""
    assert remote_control_error_type(404, "") == "NotFoundError"
    assert remote_control_error_type(403, "") == "ValidationError"


@pytest.mark.asyncio
async def test_a_mutation_answered_with_5xx_is_not_retried_and_says_so() -> None:
    calls: List[str] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(request.path)
        return rc_error("Function call failed", 500)

    async with editor(handler, max_retries=3) as session:
        with pytest.raises(UnrealError) as raised:
            await session._http_put("/remote/object/property", WRITE_TAGS)

    assert raised.value.error_type == "EngineError"
    assert calls == ["/remote/object/property"]


@pytest.mark.asyncio
async def test_a_script_timeout_says_the_script_may_still_be_running() -> None:
    async def handler(request: web.Request) -> web.Response:
        await asyncio.sleep(1.0)
        return python_reply({})

    async with editor(handler, timeout=0.2) as session:
        with pytest.raises(UnrealTimeoutError) as raised:
            await session.execute_script("import time; time.sleep(60)")

    error = raised.value
    assert isinstance(error, TimeoutError)
    assert error.error_type == "TimeoutError"
    assert "may still be running in the editor" in str(error)
    assert error.details == {"route": "/remote/object/call", "timeout_s": 0.2}


@pytest.mark.asyncio
async def test_a_read_timeout_and_a_mutation_timeout_are_told_apart() -> None:
    async def handler(request: web.Request) -> web.Response:
        await asyncio.sleep(1.0)
        return web.json_response({})

    async with editor(handler, timeout=0.2, max_retries=0) as session:
        with pytest.raises(UnrealTimeoutError) as read:
            await session.describe_object(CUBE)
        with pytest.raises(UnrealTimeoutError) as write:
            await session._http_put("/remote/object/property", WRITE_TAGS)

    assert str(read.value) == "Remote Control did not answer PUT /remote/object/describe within 0.2s"
    assert "may still apply the request" in str(write.value)


@pytest.mark.asyncio
async def test_nothing_listening_is_a_connection_error() -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    session = UnrealRuntimeSession(_settings(port, max_retries=0))
    try:
        with pytest.raises(UnrealConnectionError) as raised:
            await session.describe_object(CUBE)
    finally:
        await session.close()

    assert isinstance(raised.value, ConnectionError)
    assert raised.value.error_type == "ConnectionError"
    assert str(raised.value).startswith(f"Cannot reach Remote Control at 127.0.0.1:{port} (")
    assert raised.value.os_error is not None


@pytest.mark.asyncio
async def test_an_editor_that_drops_the_connection_may_have_crashed() -> None:
    async def handler(request: web.Request) -> web.Response:
        request.transport.close()
        raise ConnectionResetError

    async with editor(handler, max_retries=0) as session:
        with pytest.raises(UnrealConnectionError) as raised:
            await session._http_put("/remote/object/property", WRITE_TAGS)

    assert "may have crashed" in str(raised.value)


# -- U1-07: a tool script's SimulNotFound is NotFoundError ---------------------


TRACEBACK = (
    "Traceback (most recent call last):\n\n  File \"<string>\", line 75, in <module>\n\n"
    "  File \"<string>\", line 7, in actor_at\n\n{line}\n"
)


@pytest.mark.parametrize(
    "line",
    [
        f"SimulNotFound: Actor not found: {CUBE}",
        f"__main__.SimulNotFound: Actor not found: {CUBE}",
    ],
)
def test_script_not_found_reads_the_last_line(line: str) -> None:
    assert script_not_found(TRACEBACK.format(line=line)) == f"Actor not found: {CUBE}"


def test_script_not_found_ignores_other_failures() -> None:
    assert script_not_found(TRACEBACK.format(line=f"ValueError: Actor not found: {CUBE}")) is None
    assert script_not_found("") is None


@pytest.mark.asyncio
async def test_a_tool_script_that_raised_simul_not_found_is_one_not_found_shape() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {"ReturnValue": False, "CommandResult": TRACEBACK.format(line=f"SimulNotFound: Actor not found: {CUBE}")}
        )

    async with editor(handler) as session:
        result = await session.get_actor_info(CUBE)

    assert result == {"success": False, "error_type": "NotFoundError", "error": f"Actor not found: {CUBE}"}


@pytest.mark.asyncio
async def test_any_other_script_failure_stays_a_script_error() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ReturnValue": False, "CommandResult": TRACEBACK.format(line="ValueError: bad")})

    async with editor(handler) as session:
        result = await session.get_actor_info(CUBE)

    assert result["error_type"] == "ScriptError" and "ValueError: bad" in result["error"]


# -- U1-14: render settings are read back ---------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,expected",
    [
        (
            {"known": False},
            {"success": False, "applied": False, "error_type": "NotFoundError",
             "error": "Unreal has no console variable named r.DoesNotExist.Foo; nothing was changed"},
        ),
        (
            {"known": True, "previous_value": "0", "value": "0"},
            {"success": False, "applied": False, "error_type": "RenderSettingNotApplied",
             "previous_value": "0", "value": "0"},
        ),
        (
            {"known": True, "previous_value": "100", "value": "200"},
            {"success": False, "applied": False, "error_type": "RenderSettingNotApplied",
             "previous_value": "100", "value": "200"},
        ),
        ({"known": True, "previous_value": "false", "value": "true"}, {"applied": True, "value": "true"}),
        ({"known": True, "previous_value": "-1", "value": "1"}, {"applied": True, "value": "1"}),
    ],
)
async def test_a_render_setting_reports_what_unreal_holds(state: Dict[str, Any], expected: Dict[str, Any]) -> None:
    async def handler(request: web.Request) -> web.Response:
        return python_reply(state)

    async with editor(handler) as session:
        result = await session.set_render_settings("r.DoesNotExist.Foo", "1")

    assert {key: result.get(key) for key in expected} == expected
    if expected.get("error_type") == "RenderSettingNotApplied":
        reason = "kept r.DoesNotExist.Foo at 0" if state["value"] == "0" else "set r.DoesNotExist.Foo to 200"
        assert reason in result["error"]


@pytest.mark.parametrize(
    "read_back,requested,matches",
    [("0.333333", "0.3333333", True), ("75", "75.0", True), ("1e+06", "1000000", True),
     ("true", "1", True), ("false", "0", True), ("0", "1", False), ("1", "1.5", False), ("", "1", False)],
)
def test_console_values_compare_as_unreal_formats_them(read_back: str, requested: str, matches: bool) -> None:
    assert unreal_runtime._console_value_matches(read_back, requested) is matches


class _ConsoleVariables:
    """The two SystemLibrary calls the script makes, over a dict of variables."""

    def __init__(self, variables: Dict[str, str]) -> None:
        self.variables = variables
        self.commands: List[str] = []

    def get_console_variable_string_value(self, name: str) -> str:
        return self.variables.get(name, "")

    def execute_console_command(self, world: Any, command: str) -> None:
        self.commands.append(command)
        name, value = command.split(" ", 1)
        if name in self.variables:
            self.variables[name] = value


def _run_console_script(monkeypatch: pytest.MonkeyPatch, variables: Dict[str, str], name: str) -> Dict[str, Any]:
    import io
    import sys
    from contextlib import redirect_stdout
    from types import SimpleNamespace

    library = _ConsoleVariables(variables)
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace(SystemLibrary=library))
    output = io.StringIO()
    with redirect_stdout(output):
        exec(f"args = {{'name': {name!r}, 'value': '75'}}\n" + SET_CONSOLE_VARIABLE, {})
    return {"printed": json.loads(output.getvalue()), "commands": library.commands}


def test_the_script_never_runs_a_command_for_a_name_without_a_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    unknown = _run_console_script(monkeypatch, {}, "r.DumpEverything")
    known = _run_console_script(monkeypatch, {"r.ScreenPercentage": "100"}, "r.ScreenPercentage")

    assert unknown == {"printed": {"known": False}, "commands": []}
    assert known == {
        "printed": {"known": True, "previous_value": "100", "value": "75"},
        "commands": ["r.ScreenPercentage 75"],
    }
