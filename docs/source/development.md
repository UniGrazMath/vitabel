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

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting the code base,
and [semantic versioning](https://semver.org/) for the release tags.


## Changelog

A detailed changelog summarizing all changes incorporated in vitabel between
releases can be found at the
[GitHub Release Overview](https://github.com/UniGrazMath/vitabel/releases).