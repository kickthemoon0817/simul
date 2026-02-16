"""
Isaac Sim TCP socket client for remote script execution.

Connects to the stock `isaacsim.code_editor.vscode` extension's TCP socket
server (default port 8226) to execute arbitrary Python code inside a running
Isaac Sim application. No custom extension required.

Protocol:
    1. Open TCP connection to host:port
    2. Send Python code as UTF-8 bytes
    3. Receive JSON response: {"status":"ok"|"error","output":"..."}
    4. Connection closes after each execution
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScriptResult:
    """
    Result of executing a Python script inside Isaac Sim.

    Attributes:
        success: Whether execution completed without error.
        output: Captured stdout from the script.
        error_name: Exception class name if execution failed.
        error_value: Exception message if execution failed.
        traceback: Full traceback string if execution failed.
    """

    success: bool
    output: str
    error_name: str = ""
    error_value: str = ""
    traceback: str = ""


class IsaacSocketClient:
    """
    TCP socket client that sends Python code to a running Isaac Sim instance.

    Targets the stock ``isaacsim.code_editor.vscode`` extension which listens
    on ``127.0.0.1:8226`` by default. Each call opens a new TCP connection,
    sends the code, reads the JSON response, and closes.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8226,
        timeout_seconds: float = 30.0,
        read_buffer_size: int = 1024 * 1024,
    ) -> None:
        """
        Initialize the socket client.

        Args:
            host: Isaac Sim socket server host.
            port: Isaac Sim socket server port.
            timeout_seconds: Timeout for the full send+receive cycle.
            read_buffer_size: Max bytes to read per recv call.
        """
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
        self._read_buffer_size = read_buffer_size

    @property
    def address(self) -> str:
        """Return the target address as host:port string."""
        return f"{self._host}:{self._port}"

    async def execute(self, code: str) -> ScriptResult:
        """
        Execute Python code inside the running Isaac Sim process.

        Opens a TCP connection to the VS Code extension socket server,
        sends the code, waits for the JSON result, and returns it parsed.

        Args:
            code: Python source code to execute in Isaac Sim's Python scope.

        Returns:
            ScriptResult with stdout capture and error details if any.

        Raises:
            ConnectionRefusedError: If Isaac Sim is not running or the
                extension is not enabled.
            TimeoutError: If the execution exceeds timeout_seconds.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._timeout_seconds,
            )
        except (ConnectionRefusedError, OSError) as exc:
            raise ConnectionRefusedError(
                f"Cannot connect to Isaac Sim at {self.address}. "
                f"Ensure Isaac Sim is running with isaacsim.code_editor.vscode enabled."
            ) from exc

        try:
            writer.write(code.encode("utf-8"))
            await writer.drain()

            chunks: list[bytes] = []
            while True:
                chunk = await asyncio.wait_for(
                    reader.read(self._read_buffer_size),
                    timeout=self._timeout_seconds,
                )
                if not chunk:
                    break
                chunks.append(chunk)

            raw_response = b"".join(chunks).decode("utf-8")
            return self._parse_response(raw_response)

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def ping(self) -> bool:
        """
        Check if Isaac Sim is reachable by executing a trivial script.

        Returns:
            True if Isaac Sim responded successfully, False otherwise.
        """
        try:
            result = await self.execute("print('pong')")
            return result.success and "pong" in result.output
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    @staticmethod
    def _parse_response(raw: str) -> ScriptResult:
        """
        Parse the JSON response from the VS Code extension socket server.

        Args:
            raw: Raw JSON string from the server.

        Returns:
            Parsed ScriptResult.
        """
        if not raw.strip():
            return ScriptResult(success=False, output="", error_name="EmptyResponse",
                                error_value="No response from Isaac Sim")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ScriptResult(success=False, output=raw, error_name="JSONDecodeError",
                                error_value=str(exc))

        status = data.get("status", "error")
        output = data.get("output", "")

        if status == "ok":
            return ScriptResult(success=True, output=output)

        traceback_lines = data.get("traceback", [])
        traceback_str = "\n".join(traceback_lines) if isinstance(traceback_lines, list) else str(traceback_lines)

        return ScriptResult(
            success=False,
            output=output,
            error_name=data.get("ename", "UnknownError"),
            error_value=data.get("evalue", ""),
            traceback=traceback_str,
        )

