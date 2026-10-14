#!/usr/bin/env python
"""Benchmark core time-series operations of vitabel.

Not run in CI; used to track the performance of the time-series core
(construction, time conversion, hashing, windowing, saving/loading) on
high-resolution data. Prints a Markdown table.

Example::

    uv run python scripts/bench_core.py                 # 2 h @ 500 Hz
    uv run python scripts/bench_core.py --minutes 10    # quicker
    uv run python scripts/bench_core.py --zoll --output bench.md
"""

from __future__ import annotations

import argparse
import bz2
import logging
import platform
import shutil
import statistics
import tempfile
import time
import warnings

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import vitabel
from vitabel import Channel, Vitals

REPO_ROOT = Path(__file__).parent.parent


def _timeit(func: Callable[[], object], repeat: int) -> tuple[float, float]:
    """Return (median, min) wall time of ``repeat`` calls of ``func``."""
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), min(durations)


def _format_duration(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.0f} µs"


def _bench_ventilation(minutes: float, record) -> None:
    data_dir = REPO_ROOT / "examples" / "data"
    source = Vitals()
    common = dict(time_start=pd.Timestamp(1970, 1, 1), time_unit="ms")
    source.add_data_from_csv(data_dir / "flow.csv.bz2", index_col="timestamp", **common)
    source.add_data_from_csv(
        data_dir / "p1.csv.bz2",
        index_col=0,
        names=["airway_pressure", "temperature_1"],
        **common,
    )
    flow = source.get_channel("airflow")
    start = flow.first_entry
    stop = start + pd.Timedelta(minutes=minutes)
    flow = flow.truncate(start, stop)
    pressure = source.get_channel("airway_pressure").truncate(start, stop)

    recording = Vitals()
    recording.add_channel(flow)
    recording.add_channel(pressure)
    record(
        f"Vitals.compute_respiratory_phases() ({minutes:g} min)",
        lambda: recording.compute_respiratory_phases(
            flow, pressure, add_intermediate_channels=True
        ),
    )
    record(
        f"Vitals.compute_ventilation_volumes() ({minutes:g} min)",
        recording.compute_ventilation_volumes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--rate", type=float, default=500.0, help="sampling rate (Hz)")
    parser.add_argument("--repeat", type=int, default=5, help="repeats for fast ops")
    parser.add_argument(
        "--save-channels",
        type=int,
        default=4,
        help="number of channels for the multi-channel save/load benchmark",
    )
    parser.add_argument(
        "--zoll", action="store_true", help="also benchmark loading the ZOLL example"
    )
    parser.add_argument(
        "--ventilation",
        type=float,
        default=None,
        metavar="MINUTES",
        help=(
            "also benchmark respiratory phase detection and ventilation volumes "
            "on the given number of minutes of the animal lab example data"
        ),
    )
    parser.add_argument("--output", type=Path, help="also write the table to a file")
    args = parser.parse_args(argv)

    logging.getLogger("vitabel").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    n_samples = int(args.minutes * 60 * args.rate)
    rng = np.random.default_rng(42)
    seconds = np.arange(n_samples) / args.rate
    data = rng.standard_normal(n_samples)
    time_start = pd.Timestamp("2024-01-01 12:00:00")
    datetimes = (time_start + pd.to_timedelta(seconds, unit="s")).to_numpy()

    rows: list[tuple[str, float, float]] = []

    def record(name: str, func: Callable[[], object], repeat: int = 1):
        median, minimum = _timeit(func, repeat)
        rows.append((name, median, minimum))
        print(f"{name}: {_format_duration(median)}", flush=True)

    record(
        "Channel construction (float seconds + time_start)",
        lambda: Channel("ecg", seconds, data, time_start=time_start),
    )
    record(
        "Channel construction (datetime64)",
        lambda: Channel("ecg", datetimes, data),
    )

    channel = Channel("ecg", seconds, data, time_start=time_start)
    record("Channel.numeric_time()", channel.numeric_time)
    record("Channel.data_hash()", channel.data_hash)

    middle = time_start + pd.Timedelta(minutes=args.minutes / 2)
    for window in ["2s", "60s"]:
        stop = middle + pd.Timedelta(window)
        record(
            f"Channel.get_data(), {window} window",
            lambda stop=stop: channel.get_data(
                start=middle, stop=stop, include_adjacent=True
            ),
            repeat=args.repeat,
        )
        resolution = pd.Timedelta(window) / 1000
        record(
            f"Channel.get_data(), {window} window, resolution=window/1000",
            lambda stop=stop, resolution=resolution: channel.get_data(
                start=middle, stop=stop, resolution=resolution, include_adjacent=True
            ),
            repeat=args.repeat,
        )
    record(
        "Channel.get_data(), full range",
        lambda: channel.get_data(),
        repeat=args.repeat,
    )
    full_resolution = pd.Timedelta(minutes=args.minutes) / 1000
    record(
        "Channel.get_data(), full range, resolution=range/1000",
        lambda: channel.get_data(resolution=full_resolution),
        repeat=args.repeat,
    )
    record(
        "Channel.get_time_mask(), 2s window",
        lambda: channel.get_time_mask(
            start=middle, stop=middle + pd.Timedelta("2s"), include_adjacent=True
        ),
        repeat=args.repeat,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vitals.json"
        for n_channels in (1, args.save_channels):
            recording = Vitals()
            for index in range(n_channels):
                recording.add_channel(
                    Channel(f"ecg {index}", seconds, data, time_start=time_start)
                )
            record(
                f"Vitals.save_data() ({n_channels} channels)",
                lambda recording=recording: recording.save_data(path),
            )
            record(
                f"Vitals.load_data() ({n_channels} channels)",
                lambda: Vitals().load_data(path),
            )

        if args.zoll:
            zoll_path = Path(tmpdir) / "ZOLL_test_case.json"
            with (
                bz2.open(REPO_ROOT / "examples/data/ZOLL_test_case.json.bz2") as src,
                open(zoll_path, "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            record(
                "Vitals.add_defibrillator_recording() (ZOLL example)",
                lambda: Vitals().add_defibrillator_recording(zoll_path),
            )

    if args.ventilation is not None:
        _bench_ventilation(args.ventilation, record)

    header = [
        f"Benchmark: {n_samples:,} samples ({args.minutes:g} min @ {args.rate:g} Hz)",
        "",
        f"- vitabel {vitabel.__version__}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, Python {platform.python_version()}",
        f"- {platform.platform()}, {platform.processor() or platform.machine()}",
        "",
        "| Operation | median | min |",
        "|---|---|---|",
    ]
    table = header + [
        f"| {name} | {_format_duration(median)} | {_format_duration(minimum)} |"
        for name, median, minimum in rows
    ]
    print()
    print("\n".join(table))
    if args.output is not None:
        args.output.write_text("\n".join(table) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
