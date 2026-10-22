# Test Data Attribution

- `time_2ecg.npy` contains data samples taken
  from https://physionet.org/content/ecgiddb/1.0.0/Person_01/.

# Test Data

- `LP15.tar.gz`: Stryker LIFEPAK 15 recording (CodeStat export).
- `golden/`: golden files for snapshot tests (see `tests/golden.py`).
- `saved/`: files saved with older versions of vitabel (`Vitals.save_data`),
  created from `build_saveable_recording` in `tests/builders.py`.
