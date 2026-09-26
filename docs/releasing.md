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

## Policy

- **Patch** (`0.X.Y+1`): the routine bump. Do it whenever changes land on
  `main`. Untagged, no GitHub Release.
- **Minor** (`0.X+1.0`): only with a developer's explicit approval. Tagged
  and published as a GitHub Release with release notes.

## Patch bump

1. Bump all four constants in a single commit on a
   `chore/bump-version-X.Y.Z` branch. Commit subject: `chore: bump version
   to X.Y.Z` — no parentheses, no body needed.
2. Open a PR against `main` (branch protection rejects direct pushes) and
   merge it with a merge commit. Nothing else: no tag, no release.

## Minor release (developer approval required)

1. Get an explicit go-ahead from a developer for the new minor version.
2. Bump the four constants exactly as for a patch (`chore/bump-version-X.Y.0`
   branch, `chore: bump version to X.Y.0`) and merge the PR with a merge
   commit, so the bump commit keeps its SHA.
3. Tag the **bump commit itself** (not the merge commit):
   `git tag -a vX.Y.0 -m "Release vX.Y.0" <bump-sha>`, then
   `git push origin vX.Y.0` and verify with `git branch --contains vX.Y.0 main`.
4. Write release notes covering everything since the previous minor tag
   (`git log v<previous-minor>..vX.Y.0 --merges --oneline` lists the merged
   PRs): highlights, breaking or behaviour changes and anything users must do
   (for example reinstalling the Blender add-on), fixes, and the issues
   closed. Publish them:
   `gh release create vX.Y.0 --title "vX.Y.0" --notes-file notes.md --verify-tag`.

## Hard rules

- Never bump the minor (or major) version without a developer's approval.
- Never tag a patch version, and never tag a commit whose four version
  constants don't match the tag string byte-for-byte. (`v0.0.20` was once
  tagged on a commit that still said `0.0.19`.)
- Never reuse a version number whose tag is already on the remote, even if
  that tag points at the wrong commit. Skip to the next number — that's why
  the history goes `v0.0.19` → `v0.0.21`.
- Never force-move or delete a published tag.
- A minor tag without a GitHub Release (and notes) is incomplete; create the
  release in the same step.
- Tags up to `v0.1.1` predate this policy and include patch versions; leave
  them as they are.

## Marketplace

The plugin is published through
`https://github.com/kickthemoon0817/khemoo-claude-plugins`, whose
`.claude-plugin/marketplace.json` lists `simul` with
`source: { source: "github", repo: "kickthemoon0817/simul" }` and **no
version pin**. Every install or update gets whatever is on `main`.

- The version constants are advisory metadata; they don't gate what users
  receive. A push to `main` is effectively a release.
- Stale constants on `main` (the manifest claiming an old version while users
  run newer code) are a user-facing bug; routine patch bumps prevent them.
- A normal code release changes nothing in `khemoo-claude-plugins`. Edit its
  `marketplace.json` only to change this plugin's marketplace metadata
  (description, keywords, category, source repo, owner), through that repo's
  own PR flow.
