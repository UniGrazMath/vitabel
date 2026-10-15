"""Backward compatibility of the public import surface.

``tests/data/compat_surface.json`` lists the names that were importable from
the public modules before the refactoring of ``vitals.py`` / ``utils``
(generated from main @ 8964251). They must remain importable.
"""

from __future__ import annotations

import importlib
import json

from pathlib import Path

SURFACE_FILE = Path(__file__).parent / "data" / "compat_surface.json"


def test_previously_public_names_are_importable():
    surface: dict[str, list[str]] = json.loads(SURFACE_FILE.read_text())["surface"]
    missing = [
        f"{module_name}.{name}"
        for module_name, names in surface.items()
        for name in names
        if not hasattr(importlib.import_module(module_name), name)
    ]
    assert not missing, f"no longer importable: {missing}"
