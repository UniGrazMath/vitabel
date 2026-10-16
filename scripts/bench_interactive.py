"""Benchmark the interactive matplotlib plot (plot_interactive).

Drives the plot headless (IPython shell without GUI, ipympl canvas,
synthetic events) on the ZOLL example (setup of notebook 01) and reports,
per interaction, the time of the Python update and of a full Agg render
(``canvas.draw()``, which ipympl does in the kernel before sending the
image to the browser).

Uses only the public API, so it can benchmark other versions of the code:

    PYTHONPATH=<worktree>/src .venv/bin/python scripts/bench_interactive.py
"""

from __future__ import annotations

import bz2
import logging
import pathlib
import shutil
import statistics
import sys
import tempfile
import time
import warnings

from IPython.core.interactiveshell import InteractiveShell
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

ZOLL = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "data"
    / "ZOLL_test_case.json.bz2"
)


class HeadlessShell(InteractiveShell):
    def enable_gui(self, gui=None):
        pass


def load_case():
    from vitabel import Label, Vitals

    path = pathlib.Path(tempfile.mkdtemp()) / "zoll.json"
    with bz2.open(ZOLL) as source, open(path, "wb") as target:
        shutil.copyfileobj(source, target)
    case = Vitals()
    case.add_defibrillator_recording(path)
    case.find_CC_periods_acc()
    case.compute_etco2_and_ventilations()
    case.add_global_label(
        Label("ROSC", time_index=[], data=[], time_start=case.rec_start())
    )
    return case


def timed(function, repeat: int, draw: bool = True) -> tuple[float, float]:
    """Median (update, draw) times in ms; ``draw``: render the figure after
    the update (what happens after changes, not after mouse moves)."""
    updates, draws = [], []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        middle = time.perf_counter()
        if draw:
            figure.canvas.draw()
        end = time.perf_counter()
        updates.append((middle - start) * 1000)
        draws.append((end - middle) * 1000 if draw else float("nan"))
    return statistics.median(updates), statistics.median(draws)


def main() -> None:
    global figure
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    HeadlessShell.instance()
    import vitabel

    print(f"vitabel from {pathlib.Path(vitabel.__file__).parent}")
    case = load_case()

    start = time.perf_counter()
    plot = case.plot_interactive(
        channels=[["cpr_acceleration"], ["capnography"], ["ecg_pads"]],
        labels=[["ROSC", "cc_periods"], ["etco2_from_capnography", "ROSC"], ["ROSC"]],
        channel_overviews=[["cpr_acceleration"]],
        time_unit="s",
        subplots_kwargs={"figsize": (16.5, 8)},
    )
    figure = plot.center.figure
    figure.canvas.draw()
    created = (time.perf_counter() - start) * 1000
    canvas, ax = figure.canvas, figure.axes[0]

    def key(name):
        return lambda: KeyEvent("key_press_event", canvas, key=name)._process()

    def middle_of_axes():
        return sum(ax.get_xlim()) / 2, sum(ax.get_ylim()) / 2

    def click():
        px, py = ax.transData.transform(middle_of_axes())
        MouseEvent(
            "button_press_event", canvas, px, py, button=MouseButton.RIGHT
        )._process()

    rows = [("create plot + first draw (full recording)", (created, float("nan")))]
    rows.append(("pan, full recording (right)", timed(key("right"), 5)))
    for _ in range(4):  # zoom in to a ~4.5 min window
        key("+")()
    rows.append(("pan, ~4.5 min window (right)", timed(key("right"), 10)))
    rows.append(
        ("zoom in / out (+ then -)", timed(lambda: (key("+")(), key("-")()), 5))
    )
    rows.append(("annotate a point (right click)", timed(click, 10)))

    def mouse_move():
        # the move, then the redraw the browser requests (ipympl "draw" message)
        px, py = ax.transData.transform(middle_of_axes())
        MouseEvent("motion_notify_event", canvas, px, py)._process()
        canvas.handle_event({"type": "draw"})

    rows.append(("mouse move + requested redraw", timed(mouse_move, 10, draw=False)))

    print(f"{'interaction':45} {'update ms':>10} {'draw ms':>10}")
    for name, (update, draw) in rows:
        print(f"{name:45} {update:10.1f} {draw:10.1f}")


if __name__ == "__main__":
    sys.exit(main())
