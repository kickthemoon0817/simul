"""Regression test for Issue #41 — read_isaac_aovs detach AttributeError.

Issue #41 reported that ``read_isaac_aovs`` failed with
``'HydraTexture' object has no attribute 'split'`` during the
detach/cleanup phase, dropping all AOV data. The root cause: in
Kit 107.3.3 (Isaac Sim 5.1) ``rep.create.render_product`` returns
a ``HydraTexture`` wrapper, but ``ann.detach([rp])`` chains down
to ``SyntheticData._get_node_path`` which calls
``renderProductPath.split("/")`` — and ``HydraTexture`` has no
``.split``. Iter21 fixed this by extracting the path string via
``getattr(rp, "path", None) or ... or rp`` before passing to
``ann.detach([rp_path])``.

This test pins the script-construction contract: the Python
script that ``read_isaac_aovs`` ships into Isaac MUST use the
path-extraction wrapper, MUST call ``detach([rp_path])`` (not
``detach([rp])``), and MUST wrap detach + destroy in try/except
so a single detach failure does not lose collected AOV data
mid-call. A future regression that re-introduces the original
bug fails this test before it ever reaches an Isaac engine.
"""

from __future__ import annotations

import asyncio
import re

from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools


class _FakeIsaacClient:
    """Minimal IsaacSocketClient stand-in that captures executed scripts."""

    def __init__(self) -> None:
        self.executed_scripts: list[str] = []

    async def execute(self, script: str):
        self.executed_scripts.append(script)
        # Return a fake successful result. _execute_json_script reads
        # ``result.output`` (not stdout) and json.loads it. We don't
        # actually run the script — we just need read_aovs to complete
        # without raising so the captured script can be inspected.
        from types import SimpleNamespace
        return SimpleNamespace(
            success=True,
            output='{"aovs": {}, "attached": [], "camera": "/", "resolution": [256, 256], "num_frames": 1}',
            error_value=None,
            error_name=None,
            traceback=None,
        )

    async def execute_bridge_script_only(self, script: str):  # pragma: no cover
        return await self.execute(script)

    async def execute_vscode_only(self, script: str):  # pragma: no cover
        return await self.execute(script)


def _capture_script(*, aov_names=None, num_frames=5) -> str:
    """Run ``read_aovs`` against a fake client and return the executed script."""
    client = _FakeIsaacClient()
    tools = IsaacTools(client=client, settings=Settings())
    asyncio.run(
        tools.read_aovs(
            aov_names=aov_names or ["HdrColor"],
            num_frames=num_frames,
        )
    )
    assert len(client.executed_scripts) == 1, (
        f"Expected exactly one script execution, got {len(client.executed_scripts)}"
    )
    return client.executed_scripts[0]


def test_read_aovs_script_extracts_render_product_path_for_detach() -> None:
    """The script must compute ``rp_path`` from ``rp`` via getattr
    (with a fallback to ``rp`` itself for older Kit releases) before
    using it in detach. iter21 contract."""
    script = _capture_script()
    assert "rp_path" in script, (
        "Expected 'rp_path' variable in generated script — the iter21 "
        "fix for issue #41. Without it, detach receives the HydraTexture "
        "object directly and crashes on .split() in Kit 107.3.3."
    )
    # The exact extraction expression — fall through path for older Kit.
    assert "getattr(rp" in script
    assert '"path"' in script or "'path'" in script


def test_read_aovs_script_detaches_with_path_not_wrapper() -> None:
    """The detach call must use ``rp_path`` (the string), not ``rp``
    (the HydraTexture wrapper). A regression to ``detach([rp])`` would
    re-trigger issue #41 in Isaac Sim 5.1 / Kit 107.3.3."""
    script = _capture_script()
    assert "ann.detach([rp_path])" in script, (
        "Expected 'ann.detach([rp_path])' in script. If the regex below "
        "matches a 'detach([rp])' call, iter21's fix has been reverted "
        "and issue #41 will recur."
    )
    # Negative assertion: no detach([rp]) WITHOUT _path on a non-comment
    # line. A future change that introduces ann.detach([rp]) as actual
    # code would fail this test. The historical-bug-explainer comment
    # in the dedented script body still mentions the old call shape but
    # is preceded by '#', which the line-by-line scan filters out.
    bad_lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"\.detach\(\[rp(?!_path)\b[^\]]*\]\)", stripped):
            bad_lines.append(line)
    assert not bad_lines, (
        f"Found non-comment detach calls passing the HydraTexture wrapper "
        f"directly (re-introducing issue #41): {bad_lines}"
    )


def test_read_aovs_script_wraps_cleanup_in_try_except() -> None:
    """Detach AND destroy must be guarded so a single failure during
    cleanup does not lose collected AOV data. Pre-iter21 the bare
    ``ann.detach([rp])`` raised AttributeError mid-cleanup and the
    whole tool returned success=False even though all per-AOV
    statistics had already been computed.
    """
    script = _capture_script()
    # Any cleanup failure lands in the payload's ``warnings`` list and the
    # per-AOV stats already collected are returned. A regression that
    # strips the wrap will lose this marker text.
    assert "warnings.append(" in script, (
        "Expected cleanup failures to be collected into the warnings list. "
        "Without the try/except wrap, a detach failure (e.g. unexpected "
        "wrapper type in a future Kit release) would lose AOV data again."
    )
    assert '"warnings": warnings' in script


def test_read_aovs_script_prints_exactly_one_json_object() -> None:
    """The caller parses stdout as one JSON object.

    A detach warning printed on its own line used to precede the payload,
    so ``json.loads`` raised ``Extra data`` and the AOV data was dropped.
    Every warning must ride inside the single payload instead.
    """
    script = _capture_script()
    code_lines = [
        line for line in script.splitlines() if not line.strip().startswith("#")
    ]
    print_calls = [line for line in code_lines if "print(" in line]
    assert len(print_calls) == 1, print_calls
    assert "print(json.dumps(output))" in print_calls[0]


def test_read_aovs_script_preserves_attach_and_get_data_paths() -> None:
    """Sanity: the iter21 cleanup-only fix must not have touched the
    attach or get_data sections. A regression that, e.g., wraps
    attach in try/except (changing the failure semantics for
    invalid AOV names) should be caught here."""
    script = _capture_script()
    # Attach must still be in a try/except per the original design
    # (attach errors are collected into attach_errors, not raised).
    assert "ann.attach([rp])" in script  # uses rp directly — that path is fine
    assert "attach_errors" in script
    # get_data section must still iterate over annotators and store
    # per-AOV stats.
    assert "ann.get_data()" in script
    assert "results[name] = stats" in script
