"""
Instance session manager for multi-agent coordination.

Tracks which agents are using which Isaac Sim instances via file-based
session records at ``~/.simul/sessions/``. Sessions are advisory by default —
agents are informed about who's doing what but not blocked. With
``isaac_sim.enforce_claims`` enabled the MCP server turns a live claim into a
lock: mutating tools from any other agent are refused until the claim is
released or expires after ``CLAIM_TTL_SECONDS`` without a heartbeat.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SESSION_DIR = Path.home() / ".simul" / "sessions"
CLAIM_TTL_SECONDS = 120.0
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "for", "in", "on", "to", "of",
    "is", "it", "with", "from", "at", "by", "as", "this", "that",
    "isaac", "sim", "scene", "setup", "using", "test", "testing",
})


class InstanceSession:
    """
    Manages session records for a single Isaac Sim instance.

    Each instance (identified by port) has a JSON file at
    ``~/.simul/sessions/{port}.json`` containing a list of active
    sessions with their purpose, agent ID, and heartbeat timestamp.
    """

    def __init__(self, port: int, session_dir: Path | None = None) -> None:
        self._port = port
        self._dir = session_dir or _DEFAULT_SESSION_DIR
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._file = self._dir / f"{port}.json"

    def _read(self) -> list[dict[str, Any]]:
        """
        Read and return active sessions, pruning stale entries.

        Returns:
            List of session dicts with agent_id, purpose, started,
            last_active, and tools_used fields.
        """
        if not self._file.exists():
            return []
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            sessions = data.get("sessions", [])
        except (json.JSONDecodeError, OSError):
            return []
        now = time.time()
        active = [s for s in sessions if now - s.get("last_active", 0) < CLAIM_TTL_SECONDS]
        if len(active) != len(sessions):
            self._write(active)
        return active

    def _write(self, sessions: list[dict[str, Any]]) -> None:
        """Persist session list to disk."""
        try:
            with open(
                os.open(self._file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                "w", encoding="utf-8",
            ) as f:
                json.dump({"port": self._port, "sessions": sessions}, f, indent=2)
        except OSError as exc:
            logger.debug("Failed to write session file for port %d: %s", self._port, exc)

    def register(self, agent_id: str, purpose: str) -> dict[str, Any]:
        """
        Register an agent session on this instance.

        Args:
            agent_id: Unique identifier for the agent/session.
            purpose: Free-text description of what the agent is doing.

        Returns:
            Dict with registration confirmation and compatibility info.
        """
        sessions = self._read()
        existing = [s for s in sessions if s.get("agent_id") == agent_id]
        if existing:
            existing[0]["purpose"] = purpose
            existing[0]["last_active"] = time.time()
            self._write(sessions)
            return {"status": "updated", "agent_id": agent_id, "port": self._port}

        now = time.time()
        sessions.append({
            "agent_id": agent_id,
            "purpose": purpose,
            "tools_used": [],
            "started": now,
            "last_active": now,
        })
        self._write(sessions)
        return {"status": "registered", "agent_id": agent_id, "port": self._port}

    def heartbeat(self, agent_id: str, tool_name: str | None = None) -> None:
        """
        Update the heartbeat timestamp for an agent session.

        Args:
            agent_id: The agent's session identifier.
            tool_name: Optional tool name to add to tools_used list.
        """
        sessions = self._read()
        for s in sessions:
            if s.get("agent_id") == agent_id:
                s["last_active"] = time.time()
                if tool_name and tool_name not in s.get("tools_used", []):
                    tools = s.get("tools_used", [])
                    tools.append(tool_name)
                    if len(tools) > 20:
                        tools = tools[-20:]
                    s["tools_used"] = tools
                self._write(sessions)
                return

    def release(self, agent_id: str) -> dict[str, Any]:
        """
        Release an agent's session on this instance.

        Args:
            agent_id: The agent's session identifier.

        Returns:
            Dict with release confirmation.
        """
        sessions = self._read()
        before = len(sessions)
        sessions = [s for s in sessions if s.get("agent_id") != agent_id]
        self._write(sessions)
        removed = before - len(sessions)
        if not sessions:
            try:
                self._file.unlink(missing_ok=True)
            except OSError:
                pass
        return {"status": "released" if removed else "not_found", "port": self._port}

    def get_status(self) -> dict[str, Any]:
        """
        Get the current session status for this instance.

        Returns:
            Dict with session count, active sessions, and overall status.
        """
        sessions = self._read()
        return {
            "port": self._port,
            "session_count": len(sessions),
            "status": "in_use" if sessions else "free",
            "sessions": sessions,
        }


class SessionManager:
    """
    Manages instance sessions across all Isaac Sim instances.

    Provides discovery with compatibility scoring so agents can make
    informed decisions about which instance to use.
    """

    def __init__(self, session_dir: Path | None = None) -> None:
        self._dir = session_dir or _DEFAULT_SESSION_DIR
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def get_instance_session(self, port: int) -> InstanceSession:
        """
        Get the session manager for a specific instance port.

        Args:
            port: The TCP port of the Isaac Sim instance.

        Returns:
            InstanceSession for that port.
        """
        return InstanceSession(port, self._dir)

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """
        Read session status for all tracked instances.

        Returns:
            List of per-instance status dicts.
        """
        results = []
        try:
            for f in sorted(self._dir.glob("*.json")):
                try:
                    port = int(f.stem)
                except ValueError:
                    continue
                session = InstanceSession(port, self._dir)
                status = session.get_status()
                if status["sessions"]:
                    results.append(status)
        except OSError:
            pass
        return results

    def score_compatibility(
        self,
        incoming_purpose: str,
        instance_port: int,
        status: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Score how compatible an incoming agent's purpose is with an instance's
        current sessions.

        Args:
            incoming_purpose: The new agent's purpose description.
            instance_port: Port of the instance to check.
            status: Session status already read for this instance. Passing it
                    avoids re-reading and re-parsing the same session file,
                    which callers that have just called ``get_status`` would
                    otherwise do twice per instance.

        Returns:
            Dict with compatibility level, score, reason, and existing sessions.
        """
        if status is None:
            status = InstanceSession(instance_port, self._dir).get_status()
        sessions = status["sessions"]

        if not sessions:
            return {
                "port": instance_port,
                "compatibility": "clear",
                "score": 1.0,
                "reason": "No active sessions — instance is free",
                "sessions": [],
            }

        incoming_tokens = _tokenize(incoming_purpose)
        best_score = 0.0
        # With no overlap at all the reason still has to name whose work the
        # caller would be interfering with, so start from the first session.
        best_match_purpose = sessions[0].get("purpose", "")

        for s in sessions:
            existing_tokens = _tokenize(s.get("purpose", ""))
            if not incoming_tokens or not existing_tokens:
                continue
            shared = incoming_tokens & existing_tokens
            union = incoming_tokens | existing_tokens
            score = len(shared) / len(union) if union else 0.0
            if score > best_score:
                best_score = score
                best_match_purpose = s.get("purpose", "")

        if best_score > 0.5:
            level = "compatible"
            reason = f"Similar purpose: \"{best_match_purpose}\" — safe to join"
        elif best_score > 0.0:
            level = "likely_safe"
            reason = f"Some overlap with: \"{best_match_purpose}\" — different but may coexist"
        else:
            level = "caution"
            reason = f"Different purpose from: \"{best_match_purpose}\" — may cause interference"

        return {
            "port": instance_port,
            "compatibility": level,
            "score": round(best_score, 3),
            "reason": reason,
            "sessions": sessions,
        }

    def cleanup_stale(self) -> int:
        """
        Remove all stale session files.

        Returns:
            Number of stale session files cleaned up.
        """
        cleaned = 0
        try:
            for f in self._dir.glob("*.json"):
                try:
                    port = int(f.stem)
                except ValueError:
                    continue
                session = InstanceSession(port, self._dir)
                status = session.get_status()
                if not status["sessions"]:
                    try:
                        f.unlink(missing_ok=True)
                        cleaned += 1
                    except OSError:
                        pass
        except OSError:
            pass
        return cleaned


def _tokenize(text: str) -> set[str]:
    """
    Tokenize a purpose string into lowercase keywords, excluding stopwords.

    Args:
        text: Free-text purpose description.

    Returns:
        Set of meaningful lowercase tokens.
    """
    tokens = set()
    for word in text.lower().split():
        cleaned = "".join(c for c in word if c.isalnum() or c == "_")
        if cleaned and len(cleaned) > 1 and cleaned not in _STOPWORDS:
            tokens.add(cleaned)
    return tokens
