"""Snapshot ("golden") testing helpers.

Golden tests summarize the state of :class:`.Vitals` objects, channels and
labels (or other results) into JSON-compatible dictionaries and compare them
against reference files stored in ``tests/data/golden``.

Two comparison modes are supported:

``"exact"``
    For data that is loaded / parsed (deterministic): the SHA-256 hashes of
    time and data arrays need to match exactly.
``"values"``
    For computed results, which may differ in the last bits across platforms:
    hashes are ignored, and floating point values are compared with a
    tolerance.

In both modes names, lengths, keys, metadata and other non-numeric values
need to match exactly.

To (re)generate golden files, run::

    uv run python scripts/regenerate_golden.py

which runs the golden tests with the environment variable
``VITABEL_UPDATE_GOLDEN=1``. Golden files must only be regenerated if a
change in behavior is intended -- and this should be justified in the
corresponding commit message.
"""

from __future__ import annotations

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

GOLDEN_DIR: Path = Path(__file__).parent / "data" / "golden"
UPDATE_ENV_VAR = "VITABEL_UPDATE_GOLDEN"

FULL_ARRAY_LIMIT = 20_000
"""Arrays with at most this many entries are stored completely."""

N_SAMPLES = 256
"""Number of evenly spaced samples stored for larger arrays."""

RTOL = 1e-9
ATOL = 1e-12
TIME_ATOL_NS = 1_000
"""Absolute tolerance for time values (in nanoseconds) in ``"values"`` mode."""

VOLATILE_VITALS_METADATA = {"Time_of_construction", "Saving_to_json_time"}
VOLATILE_SERIES_METADATA = {"creation_date"}
"""Metadata keys containing the time of creation; their value is not compared."""

VALUES_MODE_IGNORED_KEYS = {"sha256", "first_entry", "last_entry", "timedelta"}
"""Keys that are not compared in ``"values"`` mode (hashes, and timestamps /
timedeltas formatted as strings, which are covered by the tolerance-based
comparison of the corresponding numeric time values)."""

Mode = Literal["exact", "values"]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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
        return {"timedelta": str(timedelta), "ns": int(timedelta.value)}
    if isinstance(obj, (datetime.date, np.datetime64)):
        timestamp = pd.Timestamp(obj)
        return None if pd.isna(timestamp) else str(timestamp)
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
            if obj.size <= FULL_ARRAY_LIMIT:
                return [jsonable(value) for value in obj]
            return summarize_array(obj.view(np.int64))
        if obj.size <= FULL_ARRAY_LIMIT:
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
    """Summarize a (numeric or textual) array."""
    if values is None:
        return None
    array = np.asarray(values)
    summary: dict[str, Any] = {"n": int(array.size), "dtype_kind": array.dtype.kind}

    if array.dtype.kind in "biuf":
        as_float = array.astype(np.float64)
        finite = as_float[np.isfinite(as_float)]
        summary["sha256"] = _sha256(as_float)
        summary["nan_count"] = int(np.isnan(as_float).sum())
        if finite.size > 0:
            summary["min"] = float(finite.min())
            summary["max"] = float(finite.max())
            summary["mean"] = float(finite.mean())
            summary["std"] = float(finite.std())
        if array.size <= FULL_ARRAY_LIMIT:
            summary["values"] = as_float.tolist()
        else:
            indices = np.linspace(0, array.size - 1, N_SAMPLES).astype(int)
            summary["samples"] = as_float[indices].tolist()
        return summary

    # textual / object data
    as_text = [None if _is_missing(value) else str(value) for value in array.ravel()]
    summary["sha256"] = hashlib.sha256(
        json.dumps(as_text, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if array.size <= FULL_ARRAY_LIMIT:
        summary["values"] = as_text
    return summary


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def summarize_time_index(time_index: pd.TimedeltaIndex) -> dict[str, Any]:
    """Summarize a time index (as integer nanoseconds)."""
    as_ns = pd.TimedeltaIndex(time_index).as_unit("ns").asi8
    summary: dict[str, Any] = {
        "n": int(as_ns.size),
        "unit": "ns",
        "sha256": _sha256(as_ns.astype(np.int64)),
    }
    if as_ns.size <= FULL_ARRAY_LIMIT:
        summary["values"] = as_ns.tolist()
    else:
        indices = np.linspace(0, as_ns.size - 1, N_SAMPLES).astype(int)
        summary["samples"] = as_ns[indices].tolist()
    return summary


def summarize_series(series: Channel | Label) -> dict[str, Any]:
    """Summarize a channel, label, or interval label."""
    summary: dict[str, Any] = {
        "class": type(series).__name__,
        "name": series.name,
        "len": len(series),
        "time_unit": series.time_unit,
        "time_start": None if series.time_start is None else str(series.time_start),
        "offset": str(series.offset),
        "first_entry": jsonable(series.first_entry),
        "last_entry": jsonable(series.last_entry),
        "time_index_unit": getattr(series.time_index, "unit", None),
        "time": summarize_time_index(series.time_index),
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


def _is_time_path(path: str) -> bool:
    return ".time." in path or path.endswith((".time", ".ns"))


def _compare(
    expected: Any, actual: Any, mode: Mode, path: str, mismatches: list[str]
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        if missing or extra:
            mismatches.append(
                f"{path}: keys differ (missing: {sorted(missing)}, "
                f"unexpected: {sorted(extra)})"
            )
        for key in expected.keys() & actual.keys():
            if mode == "values" and key in VALUES_MODE_IGNORED_KEYS:
                continue
            _compare(expected[key], actual[key], mode, f"{path}.{key}", mismatches)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(
                f"{path}: length differs (expected {len(expected)}, got {len(actual)})"
            )
            return
        if expected and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in expected + actual
        ):
            _compare_numeric_list(expected, actual, mode, path, mismatches)
            return
        for index, (exp, act) in enumerate(zip(expected, actual)):
            _compare(exp, act, mode, f"{path}[{index}]", mismatches)
        return

    if (
        isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        _compare_numeric_list([expected], [actual], mode, path, mismatches)
        return

    if expected != actual:
        mismatches.append(f"{path}: expected {expected!r}, got {actual!r}")


def _compare_numeric_list(
    expected: list, actual: list, mode: Mode, path: str, mismatches: list[str]
) -> None:
    exp = np.asarray(expected, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    is_stat = path.rsplit(".", 1)[-1] in {"mean", "std"}
    if mode == "exact" and not is_stat:
        equal = np.array_equal(exp, act, equal_nan=True)
    elif _is_time_path(path):
        equal = np.allclose(exp, act, rtol=0, atol=TIME_ATOL_NS, equal_nan=True)
    else:
        equal = np.allclose(exp, act, rtol=RTOL, atol=ATOL, equal_nan=True)
    if not equal:
        with np.errstate(invalid="ignore"):
            deviation = np.nanmax(np.abs(exp - act)) if exp.size else 0.0
        if exp.size == 1:
            mismatches.append(f"{path}: expected {expected[0]!r}, got {actual[0]!r}")
        else:
            mismatches.append(
                f"{path}: numeric values differ (max abs deviation {deviation!r})"
            )


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


def assert_matches_golden(name: str, summary: Any, mode: Mode = "exact") -> None:
    """Compare ``summary`` against the golden file ``tests/data/golden/<name>.json``.

    If the environment variable ``VITABEL_UPDATE_GOLDEN`` is set to ``1``,
    the golden file is (re)written instead.
    """
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
