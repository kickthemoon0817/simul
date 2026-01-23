# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.7] - Unreleased, 2026-01-14

### Added
- Added USD prim create/update/delete tooling for scene edits.

## [0.1.6] - Unreleased, 2026-01-14

### Added
- Enforced sandbox path allowlists for USD file and viewport outputs.

## [0.1.5] - Unreleased, 2026-01-14

### Added
- Exposed missing MCP tools for USD validation, mesh info, viewport, simulation, and camera control.

## [0.1.4] - Unreleased, 2026-01-14

### Added
- Implemented mesh normal computation, decimation, and OBJ export helpers.

## [0.1.3] - Unreleased, 2026-01-14

### Added
- Added rigid body tooling to enable physics and set velocities.

### Changed
- Extended Isaac Sim runtime adapter to manage rigid body state.

## [0.1.2] - Unreleased, 2026-01-14

### Added
- Added MCP tool annotations, output schemas, and rate limiting enforcement.

### Changed
- Validated tool inputs/outputs in the MCP server for compliance.

### Fixed
- Guarded tool execution with consistent schema validation.

## [0.1.1] - Unreleased, 2026-01-14

### Added
- Added SimulationApp smoke test logging that writes to `/tmp/simul_mcp_smoke.log`.

### Changed
- Renamed the Isaac Sim extension to `khemoo.simul.mcp` and updated install paths.

### Fixed
- Filled missing USD/Isaac tool details for transforms, materials, and tangents.

## [0.1.0] - Unreleased, 2025-07-31

### Added

### Changed

### Fixed
