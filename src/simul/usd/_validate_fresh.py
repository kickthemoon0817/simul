"""Validate one USD file in this fresh process, for ``validation.validate_usd_file``.

pxr keeps one in-memory copy of each layer per process, so a server process
whose loaded stages hold a file's layers cannot read that file afresh.
``validate_usd_file`` then runs this module with ``python -m``: it reads the
request (file path, ``usd`` settings, sandbox policy) pickled on stdin,
validates with nothing in memory, and writes one line to stdout,
``RESULT_MARKER`` followed by JSON: ``report`` and ``files``, or
``sandbox_denied`` with the refusal.
"""

from __future__ import annotations

import json
import pickle
import sys
from typing import Any, Dict


def main() -> int:
    from ..config import USDConfig
    from ..utils.paths import SandboxDenied
    from .sandbox import CompositionSandbox
    from .validation import RESULT_MARKER, _validate_here

    # stdin is the pipe the parent server process opened for this child and
    # wrote the request to (a PathPolicy has no JSON form); nothing else can
    # write to it.
    request = pickle.loads(sys.stdin.buffer.read())
    policy = request.get("policy")
    sandbox = CompositionSandbox(policy) if policy is not None else None
    result: Dict[str, Any]
    try:
        report, _ = _validate_here(request["file_path"], USDConfig(**request["usd"]), sandbox)
        result = {"report": report.to_dict(), "files": report.files}
    except SandboxDenied as exc:
        result = {
            "sandbox_denied": {
                "path": exc.path_str,
                "details": exc.details,
                "error_message": exc.error_message,
            }
        }
    sys.stdout.write(RESULT_MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
