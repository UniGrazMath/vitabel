#!/usr/bin/env python
"""(Re)generate the golden files in ``tests/data/golden``.

Runs all tests marked with ``golden`` with ``VITABEL_UPDATE_GOLDEN=1``, which
makes :func:`tests.golden.assert_matches_golden` write the golden files
instead of comparing against them. Additional arguments are passed to
pytest, e.g. to only regenerate a subset::

    uv run python scripts/regenerate_golden.py -k lifepak

Golden files should only be regenerated if a change in behavior is
intended; the reason should be stated in the commit message.
"""

import os
import subprocess
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main(argv: list[str]) -> int:
    env = os.environ | {"VITABEL_UPDATE_GOLDEN": "1"}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "golden",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "-q",
        *argv,
    ]
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
