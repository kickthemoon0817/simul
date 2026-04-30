# Unreal Engine E2E Checklist

**Purpose.** A structured, copy-pasteable checklist for verifying simul's
Unreal integration against a live editor. Designed to be runnable by a
human or dispatched to a subagent — each probe has a precise expected
shape so pass/fail is unambiguous.

## State of the test suite

| Layer | File | Count | Live UE needed |
|---|---|---|---|
| Adapter unit tests | `tests/unreal/test_unreal_runtime.py` | 79 | No (aiohttp mocked) |
| MCP registration | `tests/unreal/test_server_unreal_registration.py` | 2 | No |
| Setup patcher | `tests/unreal/test_setup.py` | 9 | No |
| **Live E2E (C1–C5)** | `tests/unreal/test_live.py` | 6 | Yes — auto-skip when down |

The live tier mirrors the C1–C5 probes below as `@pytest.mark.unreal_live`
tests. They auto-skip when the configured UE port doesn't answer, so
``make test`` runs cleanly on machines without UE installed.

Run modes:

```sh
.venv/bin/python -m pytest tests/unreal/         # all 96 tests; live skips if UE down
make test-unreal                                  # only the @unreal_live tier (6 tests)
.venv/bin/python -m pytest tests/unreal/test_live.py -v   # live, with UE running
```

Per-OS coverage: the live tests are OS-agnostic at the Python level
(Remote Control is HTTP). Validate on each platform by running
`make test-unreal` against a UE editor on that platform — no test is
gated on `platform.system()`.

## Prerequisites (do these once per session)

1. Pick a `.uproject`. `helloWorld` or a similar minimal project is ideal.
2. Run:
   ```sh
   .venv/bin/simul unreal setup <path>.uproject --yes
   ```
   Expected final JSON field `connected: true`, `engine_version` populated,
   `project_name` matches the file. If `connected: false`, **stop and
   diagnose** — none of the probes below will work.
3. The editor may already be running. In that case:
   ```sh
   .venv/bin/simul unreal setup <path>.uproject --no-launch --yes
   ```

## Available tools (thin-mode MCP surface)

simul ships thin mode by default — only 5 Unreal tools register. The full
~50-tool set is opt-in via registration config; this checklist targets
the thin surface because that's what agents actually see.

| Tool | Shape | Typical output size |
|---|---|---|
| `mcp__simul__unreal_health_check` | `() → {connected, engine_version, project_name, is_editor}` | ~150 B |
| `mcp__simul__ping_unreal` | `() → {reachable, latency_ms, ...}` | ~100 B |
| `mcp__simul__list_unreal_instances` | `({scan_port_start?, scan_port_end?}) → [{port, project_name, ...}]` | ~200 B × N |
| `mcp__simul__execute_unreal_script` | `({code, mode}) → {success, result}` | depends on what the script prints |
| `mcp__simul__capture_unreal_viewport` | `({resolution_x, resolution_y, format}) → {image_base64, ...}` | ≈ `w × h × 4 × 4/3` base64 bytes |

## Sanity checklist

Run in order. Each step prints a single-line verdict (PASS / FAIL / note).
**Do not proceed past a FAIL** — later probes depend on earlier ones.

### C1 — Connectivity & identity

```text
tool: mcp__simul__unreal_health_check
args: {}
pass when:
  - connected == true
  - engine_version starts with "5."
  - project_name is non-empty string
```

### C2 — Multi-instance discovery

```text
tool: mcp__simul__list_unreal_instances
args: {}
pass when:
  - response is a list with >= 1 entry
  - an entry has port == 30010 (or the port you configured)
  - project_name in that entry matches C1's project_name
```

### C3 — `execute_unreal_script` contract

The tool requires a JSON object printed to stdout. Verify both halves.

**C3.a — malformed (negative):**
```text
tool: mcp__simul__execute_unreal_script
args: {code: "print('hello')"}
pass when:
  - success == false
  - error mentions "JSON" (the tool rejects non-JSON output)
```

**C3.b — well-formed (positive):**
```text
tool: mcp__simul__execute_unreal_script
args: {code: "import unreal, json; print(json.dumps({'e': unreal.SystemLibrary.get_engine_version()}))"}
pass when:
  - success == true
  - response contains an 'e' field starting with "5."
```

### C4 — Minimal scene read

```python
# code:
import unreal, json
actors = unreal.EditorLevelLibrary.get_all_level_actors()
print(json.dumps({
    "count": len(actors),
    "first": [a.get_name() for a in actors[:5]],
}))
```

```text
pass when:
  - success == true
  - count >= 0 (any value is fine; empty map is OK)
  - first is a list of at most 5 strings
```

### C5 — Viewport capture (small)

```text
tool: mcp__simul__capture_unreal_viewport
args: {resolution_x: 256, resolution_y: 256, format: "png"}
pass when:
  - success == true
  - image_base64 is a non-empty string
  - len(image_base64) roughly proportional to 256*256 (not 1-2 bytes — means the
    HighResShot filesystem poll timed out)
note:
  - first call after editor start can take 3-5 s (shaders compile).
```

### C6 — Idempotent re-setup

```sh
.venv/bin/simul unreal setup <same .uproject> --no-launch --yes
```

```text
pass when:
  - JSON output has patched.uproject.changed == false
  - JSON output has patched.ini.changed == false (or `updated` subset only)
  - connected == true
```

## Token efficiency

Each MCP call's result goes straight into the model's context. Sloppy
probes burn tokens fast.

### Prefer granular tools over `execute_unreal_script`

In thin mode this matters less (thin mode exposes only 5 tools), but if
you enable full registration, reach for:

- `get_unreal_scene_info` instead of a Python script that walks the world
- `list_unreal_actors(max=N)` with a cap instead of unbounded list comprehensions
- `get_unreal_viewport_info` instead of a script that dumps camera state
- `capture_unreal_viewport` with native args instead of scripting `HighResShot`

### `execute_unreal_script` hygiene

- **Print exactly one JSON object.** The tool parses the last JSON line;
  multiple prints or mixed stdout/print debugging wastes tokens and can
  confuse the parser.
- **Cap list sizes.** `actors[:N]` beats the full list. 5–20 is usually
  enough for sanity; ask for more only when you know you need it.
- **Compact JSON.** `json.dumps(obj, separators=(",", ":"))` drops ~20 %
  compared to default pretty-printing — noticeable on larger payloads.
- **Don't echo inputs.** The model knows what it sent; return only the
  result. Avoid `print(json.dumps({"args": ..., "result": ...}))`.

### `capture_unreal_viewport` hygiene

- **Start small.** 256×256 is enough to verify the pipeline. PNG at 256²
  is ~20–80 KB pre-base64 → ~30–110 KB base64 in the response.
- **Full-res only when you need it.** 1920×1080 PNG is ~1–3 MB base64 —
  that will dominate your context budget for the turn.
- **JPEG for large captures.** Switch `format: "jpeg"` when visual
  fidelity is unnecessary; roughly 5–10× smaller than PNG.
- **Reuse captures.** Don't re-capture the same frame across turns — save
  the base64 once and reference it.

### Discovery hygiene

- `list_unreal_instances` with default scan range (30010–30019) is cheap.
- Widening the range to hundreds of ports multiplies the probe count
  linearly and may trip firewalls. Keep the default unless you know why.

## Running this as a subagent

A subagent can execute this checklist end-to-end. Dispatch pattern:

```text
Agent({
  description: "Unreal E2E sanity probe",
  subagent_type: "general-purpose",
  prompt: <<EOF
Run the Unreal E2E checklist at /Users/khemoo/pt/simul/docs/unreal-e2e-checklist.md
against the running editor on localhost:30010. Assume `simul unreal setup`
has already been run — do not re-run it, do not launch the editor.

Execute C1–C6 in order. For each step, emit:
  <id>: PASS|FAIL — <one-line evidence>

Bail out at the first FAIL and report the raw tool response for that step
only. Do not attempt remediation. Report in under 400 words total.

You have access to these MCP tools:
  mcp__simul__unreal_health_check
  mcp__simul__ping_unreal
  mcp__simul__list_unreal_instances
  mcp__simul__execute_unreal_script
  mcp__simul__capture_unreal_viewport

Do not use Bash to launch or kill the editor. Do not edit any files.
EOF
})
```

Keep the prompt self-contained (the subagent doesn't inherit your context)
and cap the output — the full Unreal tool return shapes can be noisy and
the summary is what matters to the parent conversation.

## Known caveats

- **Thin vs full registration.** This checklist targets the thin 5-tool
  surface that ships by default. If you've enabled the full ~50-tool
  registration, additional granular tools are available but this doc does
  not enumerate them (see `src/simul_mcp/mcp/registration/_reg_unreal.py`).
- **macOS `open -a UnrealEditor --args`** is the default launch path and
  relies on LaunchServices having registered the app. Epic Launcher
  installs register automatically; custom builds may not.
- **Linux** requires `--engine-path` on `simul unreal setup` unless
  `UnrealEditor` is on `$PATH` or `$UE_ENGINE_PATH` / `$UNREAL_ENGINE_PATH`
  points at the engine root. The CLI refuses to guess.
- **Coexisting ini sections.** `DefaultRemoteControl.ini` may contain both
  `[/Script/RemoteControlCommon.RemoteControlSettings]` (from older
  Epic docs) and `[/Script/RemoteControl.RemoteControlSettings]` (what
  the patcher writes). This is safe — each binds to a different UE
  settings class and coexistence is idempotent.
- **First viewport capture after editor start** can stall 3–5 s waiting
  for shaders; if `image_base64` comes back empty, retry once before
  declaring failure.

## Follow-ups (not in scope today)

- Wire `make test-unreal` into a CI pipeline that has a UE installation
  available (today the marker exists and runs locally; CI doesn't ship
  UE, so the live tier currently lives outside CI).
- Extend `tests/unreal/test_live.py` to cover the full ~50-tool
  registration mode beyond the thin 5-tool surface.
- Cross-link `examples/unreal/EXAMPLES.md` to this doc so readers
  starting from the example have a clear path to verification.
