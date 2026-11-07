"""Hardening for the MCP server's network transports (``--transport http`` / ``sse``).

Threat model. The server exposes tools that run arbitrary code inside the
engines (``execute_*_script``, on by default) and read or write files inside
the sandbox. Over stdio only the client that spawned the process can talk to
it. A network listener can be reached by three kinds of caller the user did
not mean to trust:

* **A web page in the user's browser.** It can send requests to
  ``http://127.0.0.1:<port>`` and, through DNS rebinding, read the answers
  under its own origin. The Host/Origin guard refuses a request whose Host
  header is not a loopback name, the bound address or a configured
  ``server.allowed_hosts`` entry (421), and a browser request whose Origin is
  not same-origin, loopback or a configured ``server.allowed_origins`` entry
  (403). It is on by default.
* **Another machine.** A non-loopback ``server.host`` (``0.0.0.0``, a LAN
  address, a host name) needs ``server.allow_public`` / ``--allow-public``;
  without it the server refuses to start. Such a bind without a bearer token
  starts with a loud warning, since the guard alone does not authenticate.
* **Another local user.** Loopback is shared by every account on the host.
  ``server.auth_token`` / ``server.auth_token_file`` make every request carry
  ``Authorization: Bearer <token>``; the token file must be private to its
  owner.

Everything here is resolved before the listener opens, so a misconfiguration
stops the server instead of serving with weaker protection than configured.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastmcp.server.auth import TokenVerifier

    from ..config import ServerConfig

# Host header names that always reach a loopback-bound listener.
LOOPBACK_HOSTS: Tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

# Shortest bearer token accepted; shorter ones are guessable over a network.
MIN_TOKEN_LENGTH = 16

# client_id of the access token a valid bearer token maps to. Every client that
# presents the token is this one principal (see SimulMCPServer._get_request_session_id).
BEARER_CLIENT_ID = "simul-bearer-token"


class HttpSecurityError(ValueError):
    """The network transport is configured in a way the server refuses to serve."""


def _bare_host(host: str) -> str:
    """Strip IPv6 brackets and surrounding space from a configured host."""
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def is_loopback_host(host: str) -> bool:
    """Return True when binding ``host`` only accepts connections from this machine.

    ``localhost`` and loopback IP literals qualify. Any other name does not,
    even one that resolves to loopback today: what it resolves to can change.

    Args:
        host: The configured bind host.

    Returns:
        Whether the bind is loopback-only.
    """
    bare = _bare_host(host).lower()
    if bare == "localhost":
        return True
    try:
        return ipaddress.ip_address(bare).is_loopback
    except ValueError:
        return False


def is_unspecified_host(host: str) -> bool:
    """Return True for the all-interfaces addresses (``0.0.0.0``, ``::``, empty)."""
    bare = _bare_host(host)
    if not bare:
        return True
    try:
        return ipaddress.ip_address(bare).is_unspecified
    except ValueError:
        return False


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def build_bearer_verifier(token: str) -> "TokenVerifier":
    """Build a FastMCP token verifier that accepts exactly ``token``.

    The comparison is constant-time over digests, unlike FastMCP's
    ``StaticTokenVerifier`` dictionary lookup.

    Args:
        token: The shared secret clients send as ``Authorization: Bearer <token>``.

    Returns:
        The verifier to install as the FastMCP server's ``auth``.
    """
    from fastmcp.server.auth import AccessToken, TokenVerifier

    expected = _token_digest(token)

    class BearerTokenVerifier(TokenVerifier):
        """Accept one static bearer token, compared in constant time."""

        async def verify_token(self, presented: str) -> Optional[AccessToken]:
            if not hmac.compare_digest(_token_digest(presented), expected):
                return None
            return AccessToken(token=presented, client_id=BEARER_CLIENT_ID, scopes=[])

    return BearerTokenVerifier()


def read_auth_token(server: "ServerConfig") -> Optional[str]:
    """Return the configured bearer token, or None when authentication is off.

    Args:
        server: The ``server`` settings section.

    Returns:
        The token from ``auth_token`` or the first line of ``auth_token_file``.

    Raises:
        HttpSecurityError: When both are set, the file is missing, unreadable,
            readable or writable by group or others, empty, or the token is
            shorter than ``MIN_TOKEN_LENGTH``.
    """
    inline = server.auth_token.get_secret_value() if server.auth_token is not None else None
    if inline is not None and server.auth_token_file:
        raise HttpSecurityError(
            "Both server.auth_token (SERVER__AUTH_TOKEN) and server.auth_token_file "
            "(SERVER__AUTH_TOKEN_FILE) are set; set one."
        )
    if server.auth_token_file:
        path = Path(os.path.expandvars(server.auth_token_file)).expanduser()
        try:
            info = path.stat()
        except OSError as exc:
            raise HttpSecurityError(f"Cannot read server.auth_token_file {path}: {exc.strerror or exc}") from None
        if not stat.S_ISREG(info.st_mode):
            raise HttpSecurityError(f"server.auth_token_file {path} is not a regular file")
        if os.name == "posix" and info.st_mode & 0o077:
            raise HttpSecurityError(
                f"server.auth_token_file {path} is accessible to group or others "
                f"(mode {stat.S_IMODE(info.st_mode):o}); run: chmod 600 {path}"
            )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise HttpSecurityError(f"Cannot read server.auth_token_file {path}: {exc}") from None
        token: Optional[str] = lines[0].strip() if lines else ""
        source = f"server.auth_token_file {path}"
    elif inline is not None:
        token = inline.strip()
        source = "server.auth_token"
    else:
        return None
    if not token:
        raise HttpSecurityError(f"{source} holds an empty token")
    if len(token) < MIN_TOKEN_LENGTH:
        raise HttpSecurityError(
            f"{source} holds a {len(token)}-character token; use at least "
            f"{MIN_TOKEN_LENGTH} (for example: python -c 'import secrets; "
            "print(secrets.token_urlsafe(32))')"
        )
    return token


@dataclass(frozen=True)
class HttpSecurity:
    """How the network transport protects itself, resolved from ``ServerConfig``.

    Attributes:
        host: The bind host.
        port: The bind port.
        public: Whether the bind accepts connections from other machines.
        host_origin_protection: Whether the Host/Origin guard is installed.
        allowed_hosts: Host header names the guard accepts on top of loopback
            and the bound address.
        allowed_origins: Browser origins the guard accepts on top of
            same-origin and loopback ones.
        auth_token: The bearer token every request must carry, or None.
        warnings: Weakened protections the operator should be told about.
    """

    host: str
    port: int
    public: bool
    host_origin_protection: bool
    allowed_hosts: Tuple[str, ...]
    allowed_origins: Tuple[str, ...]
    auth_token: Optional[str] = field(default=None, repr=False)
    warnings: Tuple[str, ...] = ()

    def run_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``FastMCP.run_async`` that install the guard.

        ``True`` is FastMCP's strict mode: every request's Host is checked
        against loopback, the bound address and ``allowed_hosts``, and every
        Origin against same-origin, loopback and ``allowed_origins``, whatever
        address the listener is bound to.
        """
        if not self.host_origin_protection:
            return {"host_origin_protection": False}
        return {
            "host_origin_protection": True,
            "allowed_hosts": list(self.allowed_hosts),
            "allowed_origins": list(self.allowed_origins),
        }

    def build_auth(self) -> Optional["TokenVerifier"]:
        """Return the FastMCP auth provider for the bearer token, or None."""
        if self.auth_token is None:
            return None
        return build_bearer_verifier(self.auth_token)

    def describe(self) -> List[str]:
        """One line each for the guard and the authentication, for the startup banner."""
        if self.host_origin_protection:
            hosts = ", ".join(self.allowed_hosts)
            guard = f"Host/Origin guard: on (hosts: {hosts}"
            if self.allowed_origins:
                guard += f"; extra origins: {', '.join(self.allowed_origins)}"
            guard += ")"
        else:
            guard = "Host/Origin guard: OFF"
        auth = "Authentication: bearer token" if self.auth_token else "Authentication: none"
        return [guard, auth]


def resolve_http_security(server: "ServerConfig") -> HttpSecurity:
    """Decide how the HTTP/SSE listener protects itself, or refuse to serve.

    Args:
        server: The ``server`` settings section.

    Returns:
        The resolved protections.

    Raises:
        HttpSecurityError: When ``server.host`` is not loopback and
            ``server.allow_public`` is off, or the bearer token is misconfigured.
    """
    host = server.host
    public = not is_loopback_host(host)
    if public and not server.allow_public:
        raise HttpSecurityError(
            f"server.host {host!r} is not a loopback address: the listener would accept "
            "connections from other machines, which could then run execute_*_script. "
            "Bind a loopback address (the default is localhost), or pass --allow-public "
            "(SERVER__ALLOW_PUBLIC=true) and set SERVER__AUTH_TOKEN_FILE to require a "
            "bearer token."
        )
    token = read_auth_token(server)

    warnings: List[str] = []
    if public and token is None:
        warnings.append(
            f"server.host {host!r} accepts connections from other machines and no "
            "bearer token is set: anyone who can reach port "
            f"{server.port} can call every tool, including execute_*_script. Set "
            "SERVER__AUTH_TOKEN_FILE, or keep the port behind a firewall."
        )
    if not server.host_origin_protection:
        warnings.append(
            "server.host_origin_protection is off: any web page the user opens can "
            "reach this server through DNS rebinding."
        )

    allowed_hosts: List[str] = list(LOOPBACK_HOSTS)
    bare = _bare_host(host)
    if not is_unspecified_host(host) and bare.lower() not in allowed_hosts:
        allowed_hosts.append(bare)
    for extra in server.allowed_hosts:
        if extra not in allowed_hosts:
            allowed_hosts.append(extra)

    return HttpSecurity(
        host=host,
        port=server.port,
        public=public,
        host_origin_protection=server.host_origin_protection,
        allowed_hosts=tuple(allowed_hosts),
        allowed_origins=tuple(server.allowed_origins),
        auth_token=token,
        warnings=tuple(warnings),
    )
