#!/usr/bin/env python
"""Compare the current output of ``make_cprdat_analysis`` with the reference
outputs of the last known-good version (see ``scripts/cprdat_reference.py``).

Prints all differences per test case (values compared with tolerance, as in
the "values" mode of the golden tests).

Usage::

    uv run python scripts/compare_cprdat.py [zoll lifepak lucas]
"""

from __future__ import annotations

import bz2
import json
import logging
import shutil
import sys
import tarfile
import tempfile
import traceback
import warnings

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.golden import GOLDEN_DIR, _compare, jsonable  # noqa: E402
from vitabel import Vitals  # noqa: E402

REFERENCE_REF = "b2b678c"


def _case_paths(tmpdir: Path) -> dict[str, Path]:
    zoll_path = tmpdir / "ZOLL_test_case.json"
    with (
        bz2.open(REPO_ROOT / "examples" / "data" / "ZOLL_test_case.json.bz2") as src,
        open(zoll_path, "wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    lifepak_dir = tmpdir / "lifepak"
    with tarfile.open(REPO_ROOT / "tests" / "data" / "LP15.tar.gz", "r:gz") as tar:
        tar.extractall(lifepak_dir, filter="data")
    return {
        "zoll": zoll_path,
        "lifepak": lifepak_dir / "LP15_Continuous.xml",
        "lucas": REPO_ROOT / "tests" / "data" / "Lucas_file_Lucas.xml",
    }


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    cases = argv or ["zoll", "lifepak", "lucas"]
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        paths = _case_paths(Path(tmp))
        for case in cases:
            reference_path = (
                GOLDEN_DIR / f"cprdat_reference_{REFERENCE_REF}_{case}.json.bz2"
            )
            with bz2.open(reference_path, "rt", encoding="utf-8") as fd:
                reference = json.load(fd)["result"]
            recording = Vitals()
            recording.add_defibrillator_recording(paths[case])
            try:
                result = recording.make_cprdat_analysis()
            except Exception:  # noqa: BLE001
                print(f"== {case}: FAILED\n{traceback.format_exc()}")
                total += 1
                continue
            actual = json.loads(json.dumps(jsonable(result)))
            mismatches: list[str] = []
            _compare(reference, actual, "values", case, mismatches)
            total += len(mismatches)
            print(f"== {case}: {len(mismatches)} difference(s)")
            for line in mismatches:
                print(f"  - {line}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
