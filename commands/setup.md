---
name: setup
description: Configure simul-mcp and choose which simulation backends to register with Claude Code (Isaac Sim / Unreal / Blender / USD-only). Triggers the simul-setup skill workflow.
---

# /simul:setup

Walk the user through configuring simul-mcp for their environment.

## What this command does

Invoke the `simul-setup` skill (`skills/simul-setup/SKILL.md`) immediately. The
skill handles:

1. **Backend selection** — ask which simulation engines the user runs (Isaac
   Sim, Unreal, Blender, USD-only) and only register those with Claude Code,
   keeping the agent's context tight.
2. **Python install** — if `simul-mcp` isn't on `PATH`, walk through
   `pip install simul-mcp` (or `uv tool install simul-mcp`).
3. **Per-backend configuration**:
   - **Unreal** — point the user at `simul unreal setup <.uproject>` for
     idempotent Remote Control config + headless launch (default).
     Reference: `CLAUDE.md` in this repo.
   - **Isaac Sim** — verify the bridge port (8226) is reachable; reference
     `config/isaac/default.yaml`.
   - **Blender** — register the runtime adapter; reference
     `src/simul_mcp/adapters/blender_runtime.py`.
   - **USD-only** — no runtime needed, headless mode.
4. **Verification** — run `simul --json info` (or per-backend health checks
   like `simul unreal health`, `simul isaac status`) and report what
   resolved cleanly vs what needs the user's attention.

## When to use

- First time installing simul on a machine.
- After enabling the simul plugin from the `khemoo` marketplace.
- After changing which simulation backends are installed.
- When troubleshooting "tools not appearing" or "MCP server not connecting"
  symptoms.

## When NOT to use

- For project-level setup (a specific `.uproject` for Unreal, etc.) — those
  flows are owned by per-engine commands like `simul unreal setup
  <.uproject>`. `/simul:setup` is the global "is simul itself working"
  command.
