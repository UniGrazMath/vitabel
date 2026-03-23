# intra-arrest_vent publish checklist

- [x] Merge `origin/main` into `intra-arrest_vent`
- [x] Resolve merge conflicts in `src/vitabel/vitals.py`

## Before publishing

- [ ] Run and fix `uv run ruff format`
- [ ] Run and fix `uv run ruff check`
- [ ] Review `compute_ventilation_volumes()` for unnecessary complexity; split into smaller helpers where useful
- [ ] Add tests for ventilation-volume outputs and reverse-airflow handling
- [ ] Add tests for repeated execution / overwrite behavior
- [ ] Clean up naming consistency for new channels/labels/plot styles
- [ ] Remove TODOs, typos, and incomplete docstrings in `src/vitabel/vitals.py`
- [ ] Verify citations/docs for new respiratory-phase and ventilation references
- [ ] Manually sanity-check derived outputs on representative data
- [ ] Re-run full test suite: `uv run pytest -q`
