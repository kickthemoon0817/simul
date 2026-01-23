# Simul-MCP Migration and Rebrand Plan

## Goal
Transfer the Isaac Sim MCP implementation into the `simul-mcp` repo, remove all legacy branding, and standardize on the new extension ID `khemoo.simul.mcp`. The main engine tool remains Isaac Sim, but the system must also support Blender, Unreal, Maya, 3ds Max, and other DCC/engine tools with bridging, scale correction, SimReady asset formatting, and per-engine feature exposure. The final goal is to provide MCP tools that let an agent understand and operate simulations like a simulation engineer and designer combined. The `isaac-mcp/` directory is reference-only and will be removed after the initial integration.

## Naming Decisions
- Extension ID: `khemoo.simul.mcp`
- Python package import namespace: `simul_mcp`
- Repo/project name: `simul-mcp`
- Publish targets: PyPI and npm

## Phase 1: Inventory and Scope
1. Identify the reference sources to import from `isaac-mcp/`:
   - MCP server and tool registration
   - Pydantic schemas
   - Tool implementations
   - Adapters (headless USD and Isaac runtime)
   - USD utilities
   - Omniverse extension (`exts/khemoo.simul.mcp`)
   - CLI entrypoints and scripts

2. Decide what becomes shared core vs. engine-specific:
   - Core: USD reader, bbox, summarize, mesh ops, MCP registry, shared schemas
   - Engine-specific: simulation control, viewport capture, camera control, engine-only APIs
   - DCC-specific: asset ingestion/export, material translation, scene graph transforms

## Phase 2: Rebrand (legacy -> simul)
1. Extension rename:
   - legacy extension path -> `exts/khemoo.simul.mcp`
   - Update `extension.toml`:
     - `name = "khemoo.simul.mcp"`
     - Replace all legacy extension keys with `exts."khemoo.simul.mcp".*`
     - Update module mappings to `khemoo/simul/mcp`
     - Update repository/description/keywords to simul branding

2. Python package rename:
   - legacy package name -> `simul_mcp`
   - Update imports across source, tests, examples, CLI

3. Docs and configs:
   - Replace all legacy branding references in README, docs, and configs
   - Update paths and extension name references

## Phase 3: Integrate into simul-mcp
1. Create `simul_mcp` package under `simul-mcp/src/`.
2. Copy and adapt core components from `isaac-mcp/src/`:
   - `mcp/` (schemas, server, tools, registry)
   - `adapters/` (headless_usd, isaac_runtime)
   - `usd/` (reader, bbox, mesh_ops, summarize)
   - `utils/` and `logging`
3. Adjust entrypoints:
   - Update CLI to `simul-mcp`
   - Update `pyproject.toml` scripts and metadata
4. Define cross-engine adapter boundaries:
   - `adapters/isaac_runtime.py` for Isaac Sim
   - `adapters/headless_usd.py` for pure USD
   - Placeholders for Blender, Unreal, Maya, 3ds Max adapters
5. Define shared bridging services:
   - Scale correction and unit normalization
   - SimReady asset format conversions
   - Scene graph translation helpers across engines

## Phase 4: Update Launchers and Config
1. Update `config/kits/isaac_sim_launcher.toml` to load `khemoo.simul.mcp`.
2. Update shell/PowerShell launchers to use new extension paths.
3. Ensure `ISAAC_SIM_PATH` workflows remain unchanged.
4. Add cross-engine config sections for Blender, Unreal, Maya, 3ds Max.

## Phase 5: Validation
1. Run lsp diagnostics on changed files.
2. Run unit tests for USD utilities.
3. If Isaac Sim is available, run smoke tests and MCP server start.
4. Add adapter-level tests for scale correction and format conversion.

## Phase 6: Cleanup
1. Remove `isaac-mcp/` after verifying all needed parts exist in `simul-mcp`.
2. Remove any lingering legacy branding references.

## Open Questions
- Confirm Python package name stays `simul_mcp` for PyPI publishing.
- Decide npm package name scope (e.g., `@khemoo/simul-mcp`).
- Identify which engine adapters must be implemented in the initial release vs. placeholders.
