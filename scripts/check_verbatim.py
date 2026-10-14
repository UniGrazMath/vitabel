#!/usr/bin/env python
"""Check that functions have been moved *verbatim*.

Compares the definition of functions (or methods) at a given git reference
with their definition in the current working tree, after normalizing away
differences that are irrelevant for a pure move:

- source formatting, comments, and line numbers (the comparison is AST-based),
- the name of the function itself,
- qualified names that are only spelled differently because of changed imports
  (configurable via ``--rename``; e.g. ``vitabel.Channel`` -> ``Channel``),
- optionally: a method that has been turned into a module-level function
  (``--method-to-function`` / ``--self-name``; for plain methods ``self`` is
  dropped or renamed, for static methods the ``@staticmethod`` decorator is
  removed; class methods are not supported), and docstrings
  (``--ignore-docstrings``).

Decorators are part of the comparison (except for the explicitly requested
method-to-function conversion). Any other difference is reported as a unified diff of the normalized code,
and the script exits with a non-zero status.

Examples
--------
Check that ``read_corpuls`` (previously in ``utils/loading.py``) is unchanged
compared to ``main``, wherever it lives now::

    uv run python scripts/check_verbatim.py --base main read_corpuls

Check a method that has been turned into a function with a new name, where
``self`` has been renamed to ``vitals``::

    uv run python scripts/check_verbatim.py --base main \\
        --self-name vitals \\
        Vitals.make_cprdat_analysis:make_cprdat_analysis

Names can be qualified with a path to disambiguate, e.g.
``src/vitabel/vitals.py::Vitals.shocks``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RENAMES: list[tuple[str, str]] = [
    ("vitabel.utils.helpers.", ""),
    ("vitabel.utils.constants.", ""),
    ("vitabel.utils.loading.", ""),
    ("vitabel.utils.", ""),
    ("vitabel.", ""),
    ("utils.helpers.", ""),
    ("utils.constants.", ""),
    ("utils.loading.", ""),
    ("utils.", ""),
    ("helpers.", ""),
    ("constants.", ""),
    ("loading.", ""),
]
"""Default dotted-name prefix rewrites, applied longest prefix first."""


@dataclass
class Definition:
    path: str
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    is_method: bool


def _git_show(root: Path, ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_ls_files(root: Path, ref: str, src_dir: str) -> list[str]:
    output = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", src_dir],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line.endswith(".py")]


def _collect_definitions(path: str, source: str) -> list[Definition]:
    tree = ast.parse(source, filename=path)
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(Definition(path, node.name, node, is_method=False))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.append(
                        Definition(
                            path, f"{node.name}.{child.name}", child, is_method=True
                        )
                    )
    return definitions


def _all_definitions(root: Path, ref: str | None, src_dir: str) -> list[Definition]:
    definitions = []
    if ref is None:
        for file in sorted((root / src_dir).rglob("*.py")):
            rel_path = file.relative_to(root).as_posix()
            definitions.extend(
                _collect_definitions(rel_path, file.read_text(encoding="utf-8"))
            )
    else:
        for rel_path in _git_ls_files(root, ref, src_dir):
            definitions.extend(
                _collect_definitions(rel_path, _git_show(root, ref, rel_path))
            )
    return definitions


def _find(definitions: list[Definition], spec: str, where: str) -> Definition:
    path = None
    if "::" in spec:
        path, spec = spec.split("::", 1)
    candidates = [
        definition
        for definition in definitions
        if (definition.qualname == spec or definition.qualname.endswith("." + spec))
        and (path is None or definition.path == path)
    ]
    if "." not in spec:
        # prefer exact (module-level) matches over methods with the same name
        exact = [d for d in candidates if d.qualname == spec]
        candidates = exact or candidates
    if not candidates:
        raise LookupError(f"{spec!r} not found ({where})")
    if len(candidates) > 1:
        locations = ", ".join(f"{d.path}::{d.qualname}" for d in candidates)
        raise LookupError(
            f"{spec!r} is ambiguous ({where}): {locations}. Qualify it as path::name."
        )
    return candidates[0]


def _dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _build_dotted(name: str, ctx: ast.expr_context) -> ast.expr:
    parts = name.split(".")
    node: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
    node.ctx = ctx
    return node


class _Normalizer(ast.NodeTransformer):
    def __init__(self, renames: list[tuple[str, str]], self_name: str | None):
        self.renames = sorted(renames, key=lambda rule: len(rule[0]), reverse=True)
        self.self_name = self_name

    def _rewrite(self, dotted: str) -> str:
        if self.self_name is not None:
            if dotted == "self":
                dotted = self.self_name
            elif dotted.startswith("self."):
                dotted = self.self_name + dotted[len("self") :]
        # apply rules until nothing changes anymore (bounded, to be safe)
        for _ in range(10):
            rewritten = self._rewrite_once(dotted)
            if rewritten == dotted:
                break
            dotted = rewritten
        return dotted

    def _rewrite_once(self, dotted: str) -> str:
        for old, new in self.renames:
            if old.endswith("."):
                if dotted.startswith(old) and len(dotted) > len(old):
                    return new + dotted[len(old) :]
            elif dotted == old:
                return new
            elif dotted.startswith(old + "."):
                return new + dotted[len(old) :]
        return dotted

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        dotted = _dotted_name(node)
        if dotted is None:
            return self.generic_visit(node)
        rewritten = self._rewrite(dotted)
        if rewritten == dotted:
            return node
        return _build_dotted(rewritten, node.ctx)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        rewritten = self._rewrite(node.id)
        if rewritten == node.id:
            return node
        return _build_dotted(rewritten, node.ctx)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if self.self_name is not None and node.arg == "self":
            node.arg = self.self_name
        return self.generic_visit(node)


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:] or [ast.Pass()]
    return body


class ConversionError(ValueError):
    """Raised if a requested method-to-function conversion is not applicable."""


def _convert_method_to_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef, self_name: str | None
) -> str | None:
    """Adapt (in place) the definition of a method for a comparison with a
    module-level function. Returns the name ``self`` is to be renamed to.

    - plain methods: the ``self`` parameter is dropped (``self_name`` is
      ``None``), or kept and renamed to ``self_name``;
    - static methods: the ``@staticmethod`` decorator is removed, the
      parameters are kept (``self_name`` must not be given);
    - class methods are not supported.
    """
    decorators = [_dotted_name(decorator) for decorator in node.decorator_list]
    if "classmethod" in decorators:
        raise ConversionError("converting a classmethod to a function is not supported")
    if "staticmethod" in decorators:
        if self_name is not None:
            raise ConversionError("a staticmethod has no 'self' parameter to rename")
        node.decorator_list = [
            decorator
            for decorator in node.decorator_list
            if _dotted_name(decorator) != "staticmethod"
        ]
        return None

    if not node.args.args or node.args.args[0].arg != "self":
        raise ConversionError("the method has no 'self' parameter")
    if self_name is None:
        node.args.args = node.args.args[1:]
        return None
    return self_name


def normalize(
    definition: Definition,
    renames: list[tuple[str, str]],
    method_to_function: str | None,
    ignore_docstrings: bool,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return a normalized copy of the function definition.

    Decorators (including ``staticmethod`` / ``classmethod``) are part of the
    comparison, except for an explicitly requested method-to-function
    conversion (``method_to_function`` not ``None``; the empty string means
    that ``self`` is dropped, otherwise it is renamed).
    """
    node = copy.deepcopy(definition.node)
    node.name = "_"

    self_name = None
    if method_to_function is not None:
        if not definition.is_method:
            raise ConversionError(
                f"{definition.qualname} is not a method, but a method-to-function "
                "conversion was requested"
            )
        self_name = _convert_method_to_function(node, method_to_function or None)

    if ignore_docstrings:
        node.body = _strip_docstring(node.body)

    node = _Normalizer(renames, self_name).visit(node)
    ast.fix_missing_locations(node)
    return node


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def compare(
    old: Definition,
    new: Definition,
    renames: list[tuple[str, str]],
    method_to_function: str | None,
    ignore_docstrings: bool,
) -> list[str]:
    """Compare two definitions; returns an empty list if they are equivalent,
    and the lines of a unified diff otherwise."""
    old_norm = normalize(old, renames, method_to_function, ignore_docstrings)
    new_norm = normalize(new, renames, None, ignore_docstrings)
    if _dump(old_norm) == _dump(new_norm):
        return []
    return list(
        difflib.unified_diff(
            ast.unparse(old_norm).splitlines(),
            ast.unparse(new_norm).splitlines(),
            fromfile=f"{old.path}::{old.qualname} (normalized)",
            tofile=f"{new.path}::{new.qualname} (normalized)",
            lineterm="",
        )
    )


def _parse_rename(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"invalid rename rule {value!r}: use OLD=NEW")
    old, new = value.split("=", 1)
    return old, new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help="function or Class.method to check; use OLD:NEW if it was renamed",
    )
    parser.add_argument(
        "--base", required=True, help="git reference with the original code"
    )
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        type=_parse_rename,
        metavar="OLD=NEW",
        help=(
            "additional dotted-name rewrite rule applied to both versions; "
            "OLD ending with '.' rewrites a prefix (e.g. 'np.=numpy.')"
        ),
    )
    parser.add_argument(
        "--no-default-renames",
        action="store_true",
        help="do not apply the default rewrite rules (vitabel./utils./... prefixes)",
    )
    parser.add_argument(
        "--method-to-function",
        action="store_true",
        help=(
            "the original is a method, the new version a function; the 'self' "
            "parameter of the original is dropped (see also --self-name)"
        ),
    )
    parser.add_argument(
        "--self-name",
        default=None,
        metavar="NAME",
        help=(
            "like --method-to-function, but 'self' is kept as the first "
            "parameter and renamed to NAME"
        ),
    )
    parser.add_argument(
        "--ignore-docstrings", action="store_true", help="do not compare docstrings"
    )
    parser.add_argument("--src", default="src", help="source directory (default: src)")
    parser.add_argument(
        "--root", default=".", help="repository root (default: current directory)"
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    renames = ([] if args.no_default_renames else list(DEFAULT_RENAMES)) + args.rename

    old_definitions = _all_definitions(root, args.base, args.src)
    new_definitions = _all_definitions(root, None, args.src)

    failures = 0
    for name in args.names:
        # split OLD:NEW on a single colon (not on the '::' path separator)
        old_spec, *rest = re.split(r"(?<!:):(?!:)", name)
        new_spec = rest[0] if rest else old_spec.split("::")[-1]
        try:
            old = _find(old_definitions, old_spec, f"at {args.base}")
            new = _find(new_definitions, new_spec, "in working tree")
        except LookupError as exc:
            print(f"ERROR {name}: {exc}")
            failures += 1
            continue

        method_to_function = args.self_name
        if method_to_function is None and args.method_to_function:
            method_to_function = ""
        location = f"{old.path}::{old.qualname} -> {new.path}::{new.qualname}"
        if method_to_function is not None and new.is_method:
            print(
                f"ERROR {location}: a method-to-function conversion was "
                "requested, but the new definition is a method"
            )
            failures += 1
            continue
        try:
            diff = compare(
                old, new, renames, method_to_function, args.ignore_docstrings
            )
        except ConversionError as exc:
            print(f"ERROR {location}: {exc}")
            failures += 1
            continue
        if diff:
            failures += 1
            print(f"DIFF  {location}")
            print("\n".join(diff))
        else:
            print(f"OK    {location}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
