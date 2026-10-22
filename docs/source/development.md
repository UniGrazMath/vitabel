# Development

Setup a development environment by using the Python project and environment [management
tool `uv`](https://docs.astral.sh/uv/). To setup the environment, simply run
```sh
uv sync
```

Package tests are contained in
[the `tests` directory](https://github.com/UniGrazMath/vitabel/tree/main/tests);
run them locally via
```sh
uv run pytest
```

## Golden tests

Some tests compare results with reference ("golden") files in
`tests/data/golden`, see `tests/golden.py` for details:

- the channels and labels created by the **loaders** (exact comparison),
- the results of the **analysis routines** (compared with a tolerance),
- the outputs of the **registry analysis** (`make_cprdat_analysis`).

They detect any change of these results. If a change is *intended* (e.g.,
a bug fix in an analysis routine), regenerate the golden files via
```sh
uv run python scripts/regenerate_golden.py
```
state the reason in the commit message, and add an entry about the changed
results to the changelog. Other behavior is covered by explicit tests;
please don't add golden files for it.

## Development scripts

The `scripts` directory contains some helpers for development:

- `scripts/bench_core.py` benchmarks core time series operations on
  high-resolution data,
- `scripts/bench_interactive.py` benchmarks the interactive plot,
- `scripts/check_verbatim.py` checks that functions have been moved between
  modules without (relevant) changes.

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting the code base,
and [semantic versioning](https://semver.org/) for the release tags.


## Architecture

Overview of the package structure. Please keep this section up to date when
adding, moving, or removing modules.

```
vitabel/
├── vitals.py                 Vitals: user-facing entry point (import, analysis, plotting)
├── timeseries.py             Channel, Label, IntervalLabel, TimeDataCollection
├── typing.py                 type aliases and small result dataclasses
├── io/                       data import
│   ├── defibrillator.py      device / export format detection and dispatch
│   ├── zoll.py               ZOLL (X Series JSON / XML, E Series / AED Pro CSV)
│   ├── lifepak.py            Stryker LIFEPAK 15 (CodeStat XML)
│   ├── lucas.py              Stryker LUCAS (CodeStat XML)
│   ├── corpuls.py            Corpuls (BDF)
│   ├── vitaldb.py            VitalDB / VitalRecorder (.vit)
│   ├── edf.py                EDF+
│   ├── eolife.py             EOlife ventilatory feedback (CSV)
│   └── tabular.py            dictionaries, DataFrames, CSV files
├── analysis/                 analysis routines
│   ├── ventilation/          capnography (etCO₂, ventilations), respiratory
│   │                         phases, ventilation volumes
│   ├── cpr/                  chest compression periods, CCF, defibrillations
│   ├── circulation/          ROSC prediction (features, pre-trained model)
│   ├── intervals.py          filtering by interval labels
│   └── threshold.py          area / duration under threshold
├── plotting/                 plotting of channels and labels
│   ├── layout.py             PlotSpec: what is plotted (subplots, time range and unit)
│   ├── data.py               data for drawing (min/max downsampling)
│   ├── interaction.py        InteractionCore: state, commands, events, undo (no rendering)
│   ├── overlays.py           FillBetween, HLine, VSpan for interactive plots
│   ├── static.py             static matplotlib plots (Channel.plot, ..., Vitals.plot)
│   ├── interactive.py        plot_interactive (entry point)
│   └── mpl/                  interactive matplotlib backend (ipympl, ipywidgets):
│                             InteractivePlot, renderer, control panel
├── registry/
│   └── cprdat.py             German Resuscitation Registry analysis
└── utils/
    ├── constants.py          channel name mappings, unit conversion factors
    ├── stylesheet.py         default plot styles
    ├── signal.py             generic signal processing helpers
    ├── serialization.py      array compression, JSON encoding
    ├── matching.py           attribute-based filtering of objects
    └── helpers.py, loading.py  backward compatible re-exports only
```

**Dependencies** point downwards: `utils` and `typing` → `timeseries` →
`io` / `analysis` / `plotting` → `vitals` → `registry`. The `plot` methods
of the time series classes delegate to `plotting` (imported when called).
Within `plotting`, `layout`, `data`, `interaction` and `overlays` depend
neither on matplotlib widgets nor on a backend; backends (`mpl`) translate
user input into `InteractionCore` commands and draw in reaction to its
events.
In particular, modules in `io`
and `analysis` must not depend on `vitals` (and must import names from the
defining modules, not via `import vitabel`); this is checked by
`tests/test_architecture.py`.

**Conventions:**

- Loading routines in `io` and analysis routines in `analysis` take
  channels, labels, or arrays and return *new*, unattached channels or
  labels (or plain data). They do not modify their inputs.
- The corresponding `Vitals` methods are thin wrappers: they resolve
  channels / labels by name, call the routine, and add or attach the
  results.
- Names that have moved remain importable from their previous location
  (`vitabel.utils`, `vitabel.utils.helpers`, `vitabel.utils.loading`);
  `tests/test_compat_imports.py` guards this public import surface.


## Changelog

Notable changes are recorded in
[`CHANGELOG.md`](https://github.com/UniGrazMath/vitabel/blob/main/CHANGELOG.md),
which is also part of this documentation ({doc}`changelog`). Pull requests
that change the behavior of the package add an entry to the topmost section
(the next release; create it if necessary). Breaking changes and changes of
results (e.g., of analysis routines) belong in their own sections, so that
users can see what they have to adapt.

## Releasing

Releases are published to [PyPI](https://pypi.org/project/vitabel/) by the
`release.yml` workflow whenever a GitHub release is published.

1. In a release preparation pull request, set the version in
   `pyproject.toml` (run `uv lock` afterwards), and give the changelog section
   of the release its version and date. Pre-releases use
   [PEP 440](https://peps.python.org/pep-0440/) versions such as `0.2.0a1`
   (alpha), `0.2.0b1` (beta), or `0.2.0rc1` (release candidate).
2. After merging, create the GitHub release from `main` with the tag
   `v<version>` (e.g. `v0.2.0a1`), using the changelog section as release
   notes. Pre-releases must be marked as such (`gh release create v0.2.0a1
   --prerelease`): they are then neither shown as the latest GitHub release
   nor installed by `pip` unless requested (`pip install --pre vitabel`).
3. The workflow checks that the tag matches the version in `pyproject.toml`
   and that the pre-release status of version and GitHub release agree, and
   publishes the package otherwise.
