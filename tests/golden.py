"""Snapshot ("golden") testing helpers.

Golden tests summarize the state of :class:`.Vitals` objects, channels and
labels (or other results) into JSON-compatible dictionaries and compare them
against reference files stored in ``tests/data/golden``.

Summaries are *self-describing*: arrays, timestamps, and timedeltas are
represented by dictionaries with a ``"kind"`` key (``"numeric_array"``,
``"text_array"``, ``"time_array"``, ``"timestamp"``, ``"timedelta"``), and the
comparison rules depend on this kind -- never on the name of a key alone.

Two comparison modes are supported:

``"exact"``
    For data that is loaded / parsed (deterministic). Everything needs to
    match exactly, including the SHA-256 hashes of complete arrays.
    The only exceptions are the (redundant) ``mean`` and ``std`` statistics of
    numeric arrays, which are compared with a tolerance as their computation
    (summation order) may differ across platforms.
``"values"``
    For computed results, which may differ in the last bits across platforms.
    Floating point values are compared with a relative tolerance
    (:data:`RTOL`, :data:`ATOL`); time values (integer nanoseconds) with an
    absolute tolerance of :data:`TIME_ATOL_NS`. Hashes of numeric and time
    arrays are not compared (the values / samples are, with tolerance).

In both modes, integers (e.g., lengths, counts, or integer metadata), strings,
booleans, ``None``, keys, and the content of text arrays (including their
hashes) must match exactly.

To (re)generate golden files, run::

    uv run python scripts/regenerate_golden.py

which runs all tests marked with ``golden`` with the environment variable
``VITABEL_UPDATE_GOLDEN=1``. Tests calling :func:`assert_matches_golden` must
be marked with ``@pytest.mark.golden`` (this is enforced). Golden files must
only be regenerated if a change in behavior is intended -- and this should be
justified in the corresponding commit message.
"""

from __future__ import annotations

import contextvars
import dataclasses
import datetime
import hashlib
import json
import os
import platform
import subprocess

from pathlib import Path, PurePath
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

import vitabel
from vitabel import Channel, IntervalLabel, Label, Vitals

DEFAULT_GOLDEN_DIR: Path = Path(__file__).parent / "data" / "golden"
GOLDEN_DIR: Path = DEFAULT_GOLDEN_DIR
UPDATE_ENV_VAR = "VITABEL_UPDATE_GOLDEN"

FULL_ARRAY_LIMIT = 20_000
"""Arrays with at most this many entries are stored completely."""

N_SAMPLES = 256
"""Number of evenly spaced samples stored for larger arrays."""

RTOL = 1e-9
ATOL = 1e-12
"""Tolerances for floating point values in ``"values"`` mode."""

TIME_ATOL_NS = 1_000
"""Absolute tolerance for time values (integer nanoseconds) in ``"values"``
mode. Times are stored as integer nanoseconds (absolute times since the
epoch), so a relative tolerance is meaningless for them; computed times are
typically derived from sampled times and float arithmetic in seconds, so
deviations well below a microsecond are considered equal."""

VOLATILE_VITALS_METADATA = {"Time_of_construction", "Saving_to_json_time"}
VOLATILE_SERIES_METADATA = {"creation_date"}
"""Metadata keys containing the time of creation; their value is not compared."""

CURRENT_TEST: contextvars.ContextVar[tuple[str, bool] | None] = contextvars.ContextVar(
    "vitabel_golden_current_test", default=None
)
"""``(node id, has golden marker)`` of the currently running test; set by an
autouse fixture in ``tests/conftest.py``."""

Mode = Literal["exact", "values"]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _evenly_spaced_indices(size: int) -> np.ndarray:
    return np.linspace(0, size - 1, N_SAMPLES).astype(int)


def jsonable(obj: Any) -> Any:
    """Convert ``obj`` into a JSON-compatible representation that is stable
    across sessions (no memory addresses, no absolute temporary paths)."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj  # NaN / inf are supported by the json module
    if isinstance(obj, (datetime.timedelta, np.timedelta64)):
        timedelta = pd.Timedelta(obj)
        if pd.isna(timedelta):
            return None
        return {"kind": "timedelta", "repr": str(timedelta), "ns": int(timedelta.value)}
    if isinstance(obj, (datetime.date, np.datetime64)):
        timestamp = pd.Timestamp(obj)
        if pd.isna(timestamp):
            return None
        return {
            "kind": "timestamp",
            "repr": str(timestamp),
            "ns": int(timestamp.as_unit("ns").value),
        }
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [jsonable(item) for item in obj]
        if isinstance(obj, (set, frozenset)):
            items = sorted(items, key=repr)
        return items
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "mM":
            return summarize_time_array(obj)
        if obj.size <= FULL_ARRAY_LIMIT and obj.dtype.kind not in "biuf":
            return jsonable(obj.tolist())
        return summarize_array(obj)
    if isinstance(obj, PurePath):
        return obj.name
    if isinstance(obj, (pd.Series, pd.Index)):
        return jsonable(obj.to_numpy())
    if isinstance(obj, pd.DataFrame):
        return {
            "columns": [str(column) for column in obj.columns],
            "index": jsonable(obj.index),
            "data": {str(column): jsonable(obj[column]) for column in obj.columns},
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__dataclass__": type(obj).__name__,
            **{
                field.name: jsonable(getattr(obj, field.name))
                for field in dataclasses.fields(obj)
            },
        }
    return f"<{type(obj).__name__}: {obj}>"


def summarize_array(values: Any) -> dict[str, Any] | None:
    """Summarize a numeric, textual, or time array."""
    if values is None:
        return None
    array = np.asarray(values)
    if array.dtype.kind in "mM":
        return summarize_time_array(array)

    if array.dtype.kind in "biuf":
        as_float = array.astype(np.float64).ravel()
        finite = as_float[np.isfinite(as_float)]
        summary: dict[str, Any] = {
            "kind": "numeric_array",
            "n": int(array.size),
            "dtype_kind": array.dtype.kind,
            "sha256": _sha256(as_float),
            "nan_count": int(np.isnan(as_float).sum()),
        }
        if finite.size > 0:
            summary["min"] = float(finite.min())
            summary["max"] = float(finite.max())
            summary["mean"] = float(finite.mean())
            summary["std"] = float(finite.std())
        if array.size <= FULL_ARRAY_LIMIT:
            summary["values"] = as_float.tolist()
        else:
            summary["samples"] = as_float[_evenly_spaced_indices(array.size)].tolist()
        return summary

    as_text = [None if _is_missing(value) else str(value) for value in array.ravel()]
    summary = {
        "kind": "text_array",
        "n": int(array.size),
        "dtype_kind": array.dtype.kind,
        "sha256": hashlib.sha256(
            json.dumps(as_text, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    if array.size <= FULL_ARRAY_LIMIT:
        summary["values"] = as_text
    return summary


def summarize_time_array(values: Any) -> dict[str, Any]:
    """Summarize an array of timestamps or timedeltas as integer nanoseconds
    (timestamps: since the epoch). Missing values (NaT) are stored as ``None``.
    """
    array = np.asarray(values)
    if array.dtype.kind == "M":
        as_ns = array.astype("datetime64[ns]").ravel()
        dtype = "datetime64"
    else:
        as_ns = array.astype("timedelta64[ns]").ravel()
        dtype = "timedelta64"
    is_nat = np.isnat(as_ns)
    as_int = as_ns.view(np.int64)
    valid = as_int[~is_nat]

    def to_list(selection: np.ndarray) -> list[int | None]:
        return [
            None if missing else int(value)
            for value, missing in zip(as_int[selection], is_nat[selection])
        ]

    summary: dict[str, Any] = {
        "kind": "time_array",
        "dtype": dtype,
        "n": int(as_int.size),
        "nat_count": int(is_nat.sum()),
        "sha256": _sha256(as_int),
    }
    if valid.size > 0:
        summary["min"] = int(valid.min())
        summary["max"] = int(valid.max())
    if as_int.size <= FULL_ARRAY_LIMIT:
        summary["values"] = to_list(np.arange(as_int.size))
    else:
        summary["samples"] = to_list(_evenly_spaced_indices(as_int.size))
    return summary


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def summarize_series(series: Channel | Label) -> dict[str, Any]:
    """Summarize a channel, label, or interval label."""
    summary: dict[str, Any] = {
        "class": type(series).__name__,
        "name": series.name,
        "len": len(series),
        "time_unit": series.time_unit,
        "time_start": jsonable(series.time_start),
        "offset": jsonable(series.offset),
        "first_entry": jsonable(series.first_entry),
        "last_entry": jsonable(series.last_entry),
        "time_index_unit": getattr(series.time_index, "unit", None),
        "time": summarize_time_array(np.asarray(series.time_index)),
        "data": summarize_array(series.data),
        "metadata": jsonable(
            {
                key: "<volatile>" if key in VOLATILE_SERIES_METADATA else value
                for key, value in series.metadata.items()
            }
        ),
        "plotstyle": jsonable(series.plotstyle),
    }
    if isinstance(series, Channel):
        summary["labels"] = [label.name for label in series.labels]
    if isinstance(series, Label):
        summary["text_data"] = summarize_array(series.text_data)
        summary["anchored_channel"] = (
            None if series.anchored_channel is None else series.anchored_channel.name
        )
        summary["plot_type"] = series.plot_type
        if not isinstance(series, IntervalLabel):
            summary["vline_text_source"] = series.vline_text_source
        summary["annotation_preset_type"] = series.annotation_preset_type
    return summary


def summarize_vitals(vitals: Vitals) -> dict[str, Any]:
    """Summarize a :class:`.Vitals` object: all channels (including their
    attached labels), all global labels, and the (non-volatile) metadata."""
    metadata = {
        key: value
        for key, value in vitals.metadata.items()
        if key not in VOLATILE_VITALS_METADATA
    }
    if "Recording_files_added" in metadata:
        metadata["Recording_files_added"] = [
            PurePath(str(path)).name for path in metadata["Recording_files_added"]
        ]
    return {
        "channels": [
            {
                **summarize_series(channel),
                "attached_labels": [
                    summarize_series(label) for label in channel.labels
                ],
            }
            for channel in vitals.channels
        ],
        "global_labels": [
            summarize_series(label) for label in vitals.data.global_labels
        ],
        "metadata": jsonable(metadata),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# How values are compared:
#   "skip"   -- not compared
#   "exact"  -- exact equality (floats included; NaN equals NaN)
#   "float"  -- floats with tolerance (RTOL / ATOL), everything else exactly
#   "time"   -- integers (nanoseconds) with tolerance TIME_ATOL_NS
#   "auto"   -- floats with tolerance in "values" mode, exactly otherwise

_NUMERIC_ARRAY_FIELDS = {"min", "max", "values", "samples"}


def _field_policy(kind: str | None, key: str, mode: Mode) -> str:
    values_mode = mode == "values"
    if kind in ("timestamp", "timedelta"):
        if key == "repr":
            return "skip" if values_mode else "exact"
        if key == "ns":
            return "time" if values_mode else "exact"
    elif kind == "numeric_array":
        if key == "sha256":
            return "skip" if values_mode else "exact"
        if key in ("mean", "std"):
            return "float"
        if key in _NUMERIC_ARRAY_FIELDS:
            return "float" if values_mode else "exact"
        return "exact"
    elif kind == "time_array":
        if key == "sha256":
            return "skip" if values_mode else "exact"
        if key in _NUMERIC_ARRAY_FIELDS:
            return "time" if values_mode else "exact"
        return "exact"
    elif kind == "text_array":
        return "exact"
    return "auto"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_equal(expected: Any, actual: Any, policy: str) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, int):
        if policy == "time":
            return abs(expected - actual) <= TIME_ATOL_NS
        return expected == actual
    # floats
    if np.isnan(expected) or np.isnan(actual):
        return bool(np.isnan(expected) and np.isnan(actual))
    if policy == "float":
        return bool(np.isclose(expected, actual, rtol=RTOL, atol=ATOL))
    return expected == actual


def _compare(
    expected: Any,
    actual: Any,
    mode: Mode,
    path: str,
    mismatches: list[str],
    policy: str = "auto",
) -> None:
    if policy == "skip":
        return
    if policy == "auto":
        policy = "float" if mode == "values" else "exact"

    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        if missing or extra:
            mismatches.append(
                f"{path}: keys differ (missing: {sorted(missing)}, "
                f"unexpected: {sorted(extra)})"
            )
        kind = expected.get("kind")
        if kind != actual.get("kind"):
            kind = None
        for key in sorted(expected.keys() & actual.keys()):
            field_policy = _field_policy(kind, key, mode)
            if field_policy == "auto":
                field_policy = policy
            _compare(
                expected[key],
                actual[key],
                mode,
                f"{path}.{key}",
                mismatches,
                policy=field_policy,
            )
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(
                f"{path}: length differs (expected {len(expected)}, got {len(actual)})"
            )
            return
        if expected and _compare_homogeneous_lists(
            expected, actual, policy, path, mismatches
        ):
            return
        differing = []
        for index, (exp, act) in enumerate(zip(expected, actual)):
            sub_mismatches: list[str] = []
            _compare(exp, act, mode, f"{path}[{index}]", sub_mismatches, policy=policy)
            if sub_mismatches:
                differing.append(index)
                if len(differing) <= 5:
                    mismatches.extend(sub_mismatches)
        if len(differing) > 5:
            mismatches.append(
                f"{path}: {len(differing)} of {len(expected)} entries differ"
            )
        return

    if _is_number(expected) and _is_number(actual):
        if not _numbers_equal(expected, actual, policy):
            mismatches.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if type(expected) is not type(actual) or expected != actual:
        mismatches.append(f"{path}: expected {expected!r}, got {actual!r}")


def _compare_homogeneous_lists(
    expected: list, actual: list, policy: str, path: str, mismatches: list[str]
) -> bool:
    """Vectorized comparison of lists that only contain floats, or only
    contain integers. Returns ``False`` if the lists are not homogeneous
    (and thus need to be compared element-wise)."""
    if all(type(value) is float for value in expected) and all(
        type(value) is float for value in actual
    ):
        exp = np.asarray(expected, dtype=np.float64)
        act = np.asarray(actual, dtype=np.float64)
        if policy == "float":
            equal = np.allclose(exp, act, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            equal = np.array_equal(exp, act, equal_nan=True)
        if not equal:
            with np.errstate(invalid="ignore"):
                deviation = np.nanmax(np.abs(exp - act))
            mismatches.append(
                f"{path}: values differ (max abs deviation {deviation!r})"
            )
        return True

    if all(type(value) is int for value in expected) and all(
        type(value) is int for value in actual
    ):
        try:
            exp = np.asarray(expected, dtype=np.int64)
            act = np.asarray(actual, dtype=np.int64)
        except OverflowError:
            return False
        differing = np.flatnonzero(exp != act)
        if policy == "time":
            differing = [
                index
                for index in differing
                if abs(expected[index] - actual[index]) > TIME_ATOL_NS
            ]
        if len(differing) > 0:
            index = differing[0]
            mismatches.append(
                f"{path}: {len(differing)} of {len(expected)} values differ "
                f"(first at [{index}]: expected {expected[index]!r}, "
                f"got {actual[index]!r})"
            )
        return True

    return False


def _provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "vitabel": vitabel.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _check_golden_marker() -> None:
    current_test = CURRENT_TEST.get()
    if GOLDEN_DIR != DEFAULT_GOLDEN_DIR or current_test is None:
        return
    node_id, has_marker = current_test
    if not has_marker:
        pytest.fail(
            f"{node_id} uses assert_matches_golden but is not marked with "
            "@pytest.mark.golden; it would be skipped by "
            "scripts/regenerate_golden.py.",
            pytrace=False,
        )


def assert_matches_golden(name: str, summary: Any, mode: Mode = "exact") -> None:
    """Compare ``summary`` against the golden file ``tests/data/golden/<name>.json``.

    If the environment variable ``VITABEL_UPDATE_GOLDEN`` is set to ``1``,
    the golden file is (re)written instead.
    """
    _check_golden_marker()
    path = GOLDEN_DIR / f"{name}.json"
    # normalize through JSON so that the comparison sees what would be stored
    summary = json.loads(json.dumps(jsonable(summary)))

    if os.environ.get(UPDATE_ENV_VAR) == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        content = {"mode": mode, "generated_with": _provenance(), "summary": summary}
        path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
        return

    if not path.exists():
        pytest.fail(
            f"Golden file {path.name} does not exist. Generate it with "
            "`uv run python scripts/regenerate_golden.py`."
        )

    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["mode"] != mode:
        pytest.fail(
            f"Golden file {path.name} was generated in mode {stored['mode']!r}, "
            f"but compared in mode {mode!r}."
        )

    mismatches: list[str] = []
    _compare(stored["summary"], summary, mode, name, mismatches)
    if mismatches:
        shown = "\n".join(f"  - {line}" for line in mismatches[:30])
        more = len(mismatches) - 30
        if more > 0:
            shown += f"\n  ... and {more} more"
        pytest.fail(
            f"Result differs from golden file {path.name} "
            f"(mode {mode!r}, generated at commit "
            f"{stored['generated_with'].get('git_commit', 'unknown')[:10]}):\n{shown}",
            pytrace=False,
        )
