<div align="center">

# Simul

**An MCP server that gives AI agents live control of Isaac Sim, Unreal Engine 5, Blender and OpenUSD.**

[![CI](https://github.com/kickthemoon0817/simul/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kickthemoon0817/simul/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-server-8A2BE2.svg)](https://modelcontextprotocol.io)

[Quickstart](#quickstart) ·
[Connect your agent](#connect-your-agent) ·
[Tools](docs/tools.md) ·
[CLI](docs/cli.md) ·
[Configuration](docs/configuration.md) ·
[Contributing](CONTRIBUTING.md)

</div>

---

Simul speaks the [Model Context Protocol](https://modelcontextprotocol.io), so
Claude Code, Codex, OpenCode or any other MCP client can inspect and edit a
running simulator or DCC tool: read the scene, change it, run physics, and
look at the result. The same operations are available from the `simul-mcp`
command line, which prints JSON when piped.

## What you can do

- **Build a scene from a prompt.** "Put a table with three cubes on it in
  Isaac Sim, give them rigid bodies, run 120 physics steps and show me the
  viewport."
- **Inspect a stage you did not write.** "Summarize this USD file: prim
  types, mesh statistics, lights, and anything with a missing material."
- **Drive an editor you already have open.** "In my open Unreal level, select
  the forklift, move it 2 m forward, and capture the level viewport."
- **Prepare assets.** "Check this Blender object's bounds and SimReady
  compliance, then export it as USD."
- **Fall back to scripts.** When no granular tool fits, the agent runs Python
  inside the editor (`execute_isaac_script`, `execute_unreal_script`,
  `execute_blender_script`). Operators can switch that off.

## Supported backends

| Backend | Versions | Transport | Needs running app? | Setup command |
|---|---|---|---|---|
| **NVIDIA Isaac Sim** | 5.1.0, 6.0.0, 6.0.1 | TCP: `khemoo.simul.mcp` bridge (8229), stock Python socket fallback (8226) | Yes | `simul-mcp isaac install-bridge` once, then `simul-mcp isaac launch` |
| **Unreal Engine 5** | 5.x | Remote Control HTTP API (30010) + `PythonScriptPlugin` | Yes (headless launch by default) | `simul unreal setup <project>.uproject --yes` |
| **Blender** | 4.2+ and 5.x | Attached: local add-on bridge in the open window. Embedded: `bpy` in the server process | Attached: yes. Embedded: no | `simul blender install-bridge`, then `simul blender attach` |
| **OpenUSD (headless)** | `usd-core` 26.3+ | In-process `pxr` | No | None |

`simul` and `simul-mcp` are the same entry point.

## Quickstart

### 1. Install

**Claude Code plugin** (installs the skills, the `/simul:setup` command, and
registers the MCP server):

```text
/plugin marketplace add kickthemoon0817/khemoo-claude-plugins
/plugin install simul@khemoo
/simul:setup
```

`/simul:setup` clones this repository to `~/.simul/source/`, installs
`simul-mcp` with `uv tool install` (falling back to `pipx` or `pip --user`),
adds the server to `~/.claude.json`, and walks you through backend selection.

**From git** (simul-mcp is not published on PyPI):

```bash
uv tool install "git+https://github.com/kickthemoon0817/simul"
# or
pip install "git+https://github.com/kickthemoon0817/simul"
```

Requires Python 3.11, 3.12 or 3.13. For embedded Blender add the `blender`
extra (`"simul-mcp[blender] @ git+https://github.com/kickthemoon0817/simul"`);
the `bpy` wheels exist only for Python 3.11 and 3.13.

### 2. Bring up a backend

**Isaac Sim**

```bash
export ISAAC_SIM_PATH=~/isaac-sim-6.0.1
simul-mcp isaac install-bridge          # once per Isaac install (add --symlink for a repo checkout)
simul-mcp isaac launch                  # starts Isaac headless with the transports enabled
simul-mcp isaac ping
```

See [docs/isaac-sim.md](docs/isaac-sim.md) for 5.x `bridge-up`, 6.0 notes and
the Docker Compose setup.

**Unreal Engine 5**

```bash
simul unreal setup /abs/path/MyProject.uproject --yes   # patches plugins + ini, launches, waits
simul unreal health
simul unreal capture viewport.png
```

See [docs/unreal-setup.md](docs/unreal-setup.md) for what setup changes, and
[docs/unreal-attachment.md](docs/unreal-attachment.md) to attach to a
specific open editor and viewport.

**Blender (attach to an open window)**

```bash
simul blender install-bridge            # builds ~/.simul/blender/simul_blender_bridge.zip
# In Blender: Preferences > Add-ons > Install from Disk, enable "Simul Blender Bridge"
simul blender attach
simul blender status
```

Then start the server with `--blender-mode attached` (or set
`BLENDER__MODE=attached`). The bridge uses protocol version 2: rebuild and
reinstall the add-on after upgrading simul. See [docs/blender-attachment.md](docs/blender-attachment.md).

**Headless USD**

```bash
simul-mcp usd info scene.usda
simul-mcp usd summary scene.usda --format json
```

### 3. Check what the server exposes

```bash
simul-mcp info              # reachable backends and registered tools
simul-mcp tools             # every MCP tool, grouped by backend
simul-mcp --json tools      # the same with descriptions and input schemas
```

## Connect your agent

Every client launches the server over stdio with `simul-mcp server`. Use the
absolute path from `which simul-mcp` when the client's environment may not
include your shell `PATH`. To run from a checkout instead, replace the
command with `uv --directory /abs/path/to/simul run simul-mcp server`.

**Claude Code**

```bash
claude mcp add --scope user simul -- simul-mcp server
```

Or commit a project-level `.mcp.json`:

```json
{
  "mcpServers": {
    "simul": { "command": "simul-mcp", "args": ["server"] }
  }
}
```

**Codex** (`~/.codex/config.toml`, or `codex mcp add simul -- simul-mcp server`)

```toml
[mcp_servers.simul]
command = "simul-mcp"
args = ["server"]
```

**OpenCode** (`~/.config/opencode/opencode.json`, or `opencode.json` in a project)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "simul": {
      "type": "local",
      "command": ["simul-mcp", "server"],
      "enabled": true
    }
  }
}
```

Keep agent context small by registering only what you use:
`simul-mcp server --backends unreal` or `--backends isaac,usd`. Append
`--config /abs/path/config.yaml` to point at your own configuration.

## Tools

| Backend | Categories | Examples |
|---|---|---|
| Headless USD | Load and validate, prim queries, meshes and bounds, edits | `load_usd_file`, `summarize_scene`, `get_bounding_box` |
| Isaac Sim | Scene inspection, prim editing, physics, simulation control, materials, cameras and viewport, rendering and AOVs, OmniGraph, extensions, stage I/O, UI state, logs, multi-instance | `create_isaac_object`, `step_isaac_simulation`, `capture_isaac_viewport` |
| Unreal Engine | Thin set by default: health, instances, named editor controls, capture, scripting. `--unreal-tools full` adds actors, physics, materials, meshes, PIE, USD import/export | `control_unreal_ui`, `capture_unreal_viewport`, `execute_unreal_script` |
| Blender | Objects, materials, rigid bodies and constraints, modifiers, animation, baking, viewport capture, UI controls | `create_blender_object`, `bake_blender_simulation`, `capture_blender_viewport` |
| SimReady | Metadata, hierarchy, compliance checks, USD export | `validate_simready_compliance`, `export_simready_usd` |
| Server | Usage statistics | `get_tool_usage_stats` |

`simul-mcp tools` lists every tool for the backends you select (no engine
needed); add `--json` for descriptions and input schemas. `--unreal-tools` and
`--blender-tools` pick a thin or full surface. The full catalog is in
[docs/tools.md](docs/tools.md).

## Documentation

| Guide | Contents |
|---|---|
| [docs/cli.md](docs/cli.md) | Every `simul-mcp` command and its main flags |
| [docs/configuration.md](docs/configuration.md) | YAML config, `SECTION__KEY` environment overrides, security switches, logging |
| [docs/tools.md](docs/tools.md) | MCP tool catalog by backend |
| [docs/isaac-sim.md](docs/isaac-sim.md) | Bridge install, launch, 6.0 differences, containers, transport behaviour |
| [docs/unreal-setup.md](docs/unreal-setup.md) | What `simul unreal setup` configures, cross-host flags |
| [docs/unreal-attachment.md](docs/unreal-attachment.md) | Attach to an existing Unreal editor, named controls, agent overlay |
| [docs/unreal-e2e-checklist.md](docs/unreal-e2e-checklist.md) | Live verification probes for Unreal |
| [docs/blender-attachment.md](docs/blender-attachment.md) | Attach to an existing Blender window |
| [docs/development.md](docs/development.md) | Repository layout, architecture, tests, formatting |

Example galleries with prompts and results:
[Blender](examples/blender/EXAMPLES.md) and [Unreal](examples/unreal/EXAMPLES.md).

The Isaac Sim scripting reference ships inside the package and is served to
agents as the MCP resource `simul://isaac-sim/skills`. The Claude Code plugin
also installs task skills from [`skills/`](skills).

## Security

Every backend can run agent-written Python inside the target application, and
none of the transports authenticate by default:

- Unreal Remote Control with remote Python execution has no authentication.
  `simul unreal setup` keeps UE's default loopback binding. Binding to another
  interface requires `--bind <host> --allow-public`, optionally with
  `--passphrase`; read [docs/unreal-setup.md](docs/unreal-setup.md) first.
- The Isaac Sim bridge and Python socket bind to the local machine. Run
  `simul-mcp` on the same host; cross-host Isaac is not supported.
- `SECURITY__ALLOW_SCRIPT_EXECUTION=false` removes the arbitrary-code tools
  from the MCP server. See [docs/configuration.md](docs/configuration.md#security).

To report a vulnerability, open an issue asking for a private contact and
leave the exploit details out of the public thread.

## Contributing

Bug reports and pull requests are welcome. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, test tiers and PR
conventions, and file issues at
<https://github.com/kickthemoon0817/simul/issues>.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

- The NVIDIA Isaac Sim team for the simulation platform
- Pixar and the OpenUSD community for USD
- Epic Games for Unreal Engine's Remote Control and Python APIs
- The Blender Foundation
- The Model Context Protocol community for the specification and FastMCP
