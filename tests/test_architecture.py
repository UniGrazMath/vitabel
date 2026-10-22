"""Checks the intended dependency structure of the package.

Modules in ``vitabel.io`` and ``vitabel.analysis`` operate on channels and
labels. They must not depend on the high-level modules (``vitals``,
``registry``), and they import names from the defining modules
(``from vitabel.timeseries import Channel``) instead of accessing them via
the package at runtime (``vitabel.Channel``), which invites import cycles.
Imports in ``if TYPE_CHECKING:`` blocks are exempt.
"""

from __future__ import annotations

import ast

from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src" / "vitabel"
FORBIDDEN_MODULES = ("vitabel.vitals", "vitabel.registry")


def _runtime_imports(node: ast.AST):
    """Import statements outside of ``if TYPE_CHECKING:`` blocks."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
            for orelse in child.orelse:
                yield from _runtime_imports(orelse)
            continue
        yield from _runtime_imports(child)


def _violations(source: str) -> list[str]:
    violations = []
    for node in _runtime_imports(ast.parse(source)):
        statement = ast.unparse(node)
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                violations.append(f"relative import: {statement}")
            elif node.module == "vitabel":
                violations.append(f"import from the top-level package: {statement}")
            elif node.module.startswith(FORBIDDEN_MODULES):
                violations.append(f"forbidden module: {statement}")
        elif any(alias.name.split(".")[0] == "vitabel" for alias in node.names):
            violations.append(f"binds the package object: {statement}")
    return violations


MODULES = sorted(
    path
    for subpackage in ["io", "analysis"]
    for path in (SRC_DIR / subpackage).rglob("*.py")
)


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda path: str(path.relative_to(SRC_DIR))
)
def test_io_and_analysis_do_not_depend_on_higher_layers(path):
    assert not _violations(path.read_text())
