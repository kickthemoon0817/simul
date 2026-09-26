# Releasing simul

## Version constants

The version lives in **four places that always move together**:

- `pyproject.toml` → `[project] version = "X.Y.Z"`
- `.claude-plugin/plugin.json` → `"version": "X.Y.Z"`
- `src/simul_mcp/__init__.py` → `__version__ = "X.Y.Z"`
- `src/simul_mcp/bridge_ext/khemoo.simul.mcp/config/extension.toml` →
  `[package] version = "X.Y.Z"`. The Isaac Sim bridge extension ships inside
  the wheel; its version-suffixed Kit ID (`khemoo.simul.mcp-X.Y.Z`) must
  match the package so either side reports the same version.

## Release sequence — do not deviate

1. Bump all four constants in a single commit on a
   `chore/bump-version-X.Y.Z` branch. Commit subject: `chore: bump version
   to X.Y.Z` — no parentheses, no body needed.
2. Open a PR against `main` (branch protection rejects direct pushes). Merge
   with a merge commit so the bump commit keeps its SHA for the tag.
3. After the PR merges, fetch and tag the **bump commit itself** (not the
   merge commit) with an annotated tag:
   `git tag -a vX.Y.Z -m "Release vX.Y.Z" <bump-sha>`.
4. Push the tag: `git push origin vX.Y.Z`, then verify it is reachable from
   `main`: `git branch --contains vX.Y.Z main`.

## Hard rules

- Never tag a commit whose four version constants don't match the tag
  string byte-for-byte. (`v0.0.20` was once tagged on a commit that still
  said `0.0.19`.)
- Never reuse a version number whose tag is already on the remote, even if
  that tag points at the wrong commit. Skip to the next number — that's why
  the history goes `v0.0.19` → `v0.0.21`.
- Never force-move or delete a published tag.
- Pre-1.0 semantics are loose: bias toward a minor bump (`0.X.0`) when
  meaningful features land and a patch bump (`0.X.Y`) for fixes only. Audit
  `git log v<last-tag>..HEAD --oneline` before picking the number.

## Marketplace

The plugin is published through
`https://github.com/kickthemoon0817/khemoo-claude-plugins`, whose
`.claude-plugin/marketplace.json` lists `simul` with
`source: { source: "github", repo: "kickthemoon0817/simul" }` and **no
version pin**. Every install or update gets whatever is on `main`.

- The version constants are advisory metadata; they don't gate what users
  receive. A push to `main` is effectively a release.
- Stale constants on `main` (the manifest claiming an old version while users
  run newer code) are a user-facing bug; the sequence above prevents them.
- A normal code release changes nothing in `khemoo-claude-plugins`. Edit its
  `marketplace.json` only to change this plugin's marketplace metadata
  (description, keywords, category, source repo, owner), through that repo's
  own PR flow.
