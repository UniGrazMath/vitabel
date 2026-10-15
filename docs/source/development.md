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

Many tests compare the results of data loading and analysis routines against
reference ("golden") files stored in `tests/data/golden`, see `tests/golden.py`
for details. If a change in behavior is *intended*, the golden files can be
regenerated via
```sh
uv run python scripts/regenerate_golden.py
```
Please state the reason for regenerating golden files in the commit message.

## Development scripts

The `scripts` directory contains some helpers for development:

- `scripts/bench_core.py` benchmarks core time series operations on
  high-resolution data,
- `scripts/check_verbatim.py` checks that functions have been moved between
  modules without (relevant) changes,
- `scripts/cprdat_reference.py` (re)generates reference outputs for the
  registry analysis (`Vitals.make_cprdat_analysis`).

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
`io` / `analysis` → `vitals` → `registry`. In particular, modules in `io`
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

A detailed changelog summarizing all changes incorporated in vitabel between
releases can be found at the
[GitHub Release Overview](https://github.com/UniGrazMath/vitabel/releases).