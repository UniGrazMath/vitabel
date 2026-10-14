# Test Data Attribution

- `time_2ecg.npy` contains data samples taken
  from https://physionet.org/content/ecgiddb/1.0.0/Person_01/.

# Test Data

- `LP15.tar.gz`: Stryker LIFEPAK 15 recording (CodeStat export).
- `golden/`: golden files for snapshot tests (see `tests/golden.py`), and
  reference outputs of the registry analysis (`cprdat_reference_*.json.bz2`,
  see `scripts/cprdat_reference.py`).
