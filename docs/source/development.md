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

### Golden tests

Many tests compare the results of data loading and analysis routines against
reference ("golden") files stored in `tests/data/golden`, see `tests/golden.py`
for details. If a change in behavior is *intended*, the golden files can be
regenerated via
```sh
uv run python scripts/regenerate_golden.py
```
Please state the reason for regenerating golden files in the commit message.

### Development scripts

The `scripts` directory contains some helpers for development:

- `scripts/bench_core.py` benchmarks core time series operations on
  high-resolution data,
- `scripts/check_verbatim.py` checks that functions have been moved between
  modules without (relevant) changes,
- `scripts/cprdat_reference.py` (re)generates reference outputs for the
  registry analysis (`Vitals.make_cprdat_analysis`).

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting the code base,
and [semantic versioning](https://semver.org/) for the release tags.


## Changelog

A detailed changelog summarizing all changes incorporated in vitabel between
releases can be found at the
[GitHub Release Overview](https://github.com/UniGrazMath/vitabel/releases).