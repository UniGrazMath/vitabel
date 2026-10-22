# Changelog

Notable changes to `vitabel`, newest first.

## [0.2.0a1] - 2026-10-22

First pre-release of 0.2: the package has been restructured, the time series
core is much faster, and the interactive plot has been rebuilt. Install it with
`pip install --pre vitabel`.

### Breaking changes

- Requires Python ≥ 3.11 ([#296](https://github.com/UniGrazMath/vitabel/pull/296))
  and pandas ≥ 3.0 ([#304](https://github.com/UniGrazMath/vitabel/pull/304)).
- Interactive plot: redrawing no longer clears the axes, so your own changes to
  the figure persist. If you re-applied them in a matplotlib `draw_event`
  handler, remove it; for content that should follow the data, use
  `plot.add_overlay(...)` and `plot.on("labels_changed", ...)` (see the guide
  *Interactive plots*) ([#302](https://github.com/UniGrazMath/vitabel/pull/302)).
- `Vitals.rec_start()` / `rec_stop()` return the first / last sample of the
  non-empty channels (previously the earliest `time_start`), and `None` for
  recordings without data ([#303](https://github.com/UniGrazMath/vitabel/pull/303)).
- `Vitals.compute_etco2_and_ventilations` raises a `TypeError` for unknown
  keyword arguments instead of ignoring them
  ([#303](https://github.com/UniGrazMath/vitabel/pull/303)).
- `IntervalLabel.contains_time` raises a `TypeError` for time zone aware
  timestamps ([#304](https://github.com/UniGrazMath/vitabel/pull/304)).
- Removed `Vitals.dats` ([#297](https://github.com/UniGrazMath/vitabel/pull/297))
  and names that were importable only by accident, such as
  `vitabel.vitals.DataSlice` (use `vitabel.typing.DataSlice`) or
  `vitabel.utils.pd`. All functions and constants that were importable from
  `vitabel.utils` still are ([#298](https://github.com/UniGrazMath/vitabel/pull/298)).

### Changed results

Re-running an analysis on the same data can give different results than with
0.1.1:

- VitalDB recordings use the time zone of the recording instead of the one of
  the computer; labels saved from earlier imports can be shifted by the
  difference ([#297](https://github.com/UniGrazMath/vitabel/pull/297)).
- Chest compression periods from the accelerometer (`find_CC_periods_acc`)
  change for recordings with gaps or irregular sampling, and with them the
  ROSC prediction and `make_cprdat_analysis`
  ([#303](https://github.com/UniGrazMath/vitabel/pull/303)).
- ROSC predictions are reproducible (they were partly random)
  ([#299](https://github.com/UniGrazMath/vitabel/pull/299)).
- Plots: the series of `Vitals.plot` share one time axis
  ([#241](https://github.com/UniGrazMath/vitabel/issues/241)); with a
  `resolution`, channels show the minimum and maximum per interval, so peaks
  stay visible ([#302](https://github.com/UniGrazMath/vitabel/pull/302)).

### Added

- Loaders and analyses are also available as functions on channels in the new
  modules `vitabel.io`, `vitabel.analysis`, `vitabel.plotting`, and
  `vitabel.registry` (e.g. `detect_etco2_and_ventilations`, `predict_rosc`);
  the `Vitals` methods work as before
  ([#298](https://github.com/UniGrazMath/vitabel/pull/298)).
- Interactive plot: undo / redo (`ctrl+z` / `ctrl+y`), events
  (`plot.on(...)`), `plot.set_window(start, stop)`, and overlays that update
  with the data (`FillBetween`, `HLine`, `VSpan` from `vitabel.plotting`)
  ([#302](https://github.com/UniGrazMath/vitabel/pull/302)).
- `load_data` warns if the data in a file has been modified outside of
  `vitabel` ([#297](https://github.com/UniGrazMath/vitabel/pull/297)).
- Documentation: a guide to interactive plots, label plot types
  ([#268](https://github.com/UniGrazMath/vitabel/issues/268)), an architecture
  overview, and this changelog.

### Performance

- Time series operations on large recordings are 10 to 500 times faster
  (creating channels, `get_data`, `save_data` / `load_data`);
  `compute_ventilation_volumes` on 8 minutes of animal lab data takes 1 s
  instead of 173 s ([#301](https://github.com/UniGrazMath/vitabel/pull/301)).
- The interactive plot opens about 18 times faster (ZOLL example: 0.2 s
  instead of 3.9 s) ([#302](https://github.com/UniGrazMath/vitabel/pull/302)).

### Fixed

- `make_cprdat_analysis` works again (it failed since 0.1.0), and includes the
  chest compression analysis for LIFEPAK and LUCAS recordings
  ([#299](https://github.com/UniGrazMath/vitabel/pull/299)).
- Loading: `read_zollcsv` failed for recordings without CPR
  ([#277](https://github.com/UniGrazMath/vitabel/pull/277));
  `add_defibrillator_recording` and the VitalDB import modified the passed
  `metadata` ([#297](https://github.com/UniGrazMath/vitabel/pull/297),
  [#298](https://github.com/UniGrazMath/vitabel/pull/298)).
- Time data: a `pd.Series` with a non-default index, a leading `NaT`, and
  multi-dimensional arrays were handled wrongly
  ([#288](https://github.com/UniGrazMath/vitabel/pull/288),
  [#303](https://github.com/UniGrazMath/vitabel/pull/303),
  [#304](https://github.com/UniGrazMath/vitabel/pull/304)).
- `rec_start` / `rec_stop` failed for recordings with empty channels and for
  relative recordings; `IntervalLabel.contains_time` failed for pandas
  timestamps ([#303](https://github.com/UniGrazMath/vitabel/pull/303)).
- Analysis: `compute_ventilation_volumes` failed with an
  `OutOfBoundsTimedelta` ([#286](https://github.com/UniGrazMath/vitabel/pull/286));
  CC period and ROSC detection failed for some relative recordings; `deriv` and
  `integrate` were wrong for irregular sampling; the legacy keyword
  `breaththresh` always raised; `cycle_duration_analysis` picked `cc` or
  `cc_depth` at random ([#303](https://github.com/UniGrazMath/vitabel/pull/303)).
- Static plots: misaligned series
  ([#241](https://github.com/UniGrazMath/vitabel/issues/241)); relative
  recordings not starting at 0 were not shown;
  `IntervalLabel.plot(plot_type="hline")` failed; channels in milliseconds
  were not downsampled; collections without channels could not be plotted
  ([#302](https://github.com/UniGrazMath/vitabel/pull/302)); scatter labels
  with all values 0 were not drawn
  ([#303](https://github.com/UniGrazMath/vitabel/pull/303)).
- Interactive plot: annotations lost sub-second precision
  ([#279](https://github.com/UniGrazMath/vitabel/pull/279),
  [#280](https://github.com/UniGrazMath/vitabel/pull/280)); "Apply" in the
  settings reset the y limits; the marker of a pending interval was missing in
  other subplots ([#255](https://github.com/UniGrazMath/vitabel/issues/255))
  ([#302](https://github.com/UniGrazMath/vitabel/pull/302)).

### Development

- Snapshot tests for loaders, analyses, and registry outputs; tests for
  loading files saved with older versions; headless tests of the interactive
  plot ([#297](https://github.com/UniGrazMath/vitabel/pull/297),
  [#302](https://github.com/UniGrazMath/vitabel/pull/302),
  [#307](https://github.com/UniGrazMath/vitabel/pull/307)).
- A CI lint job, and a release workflow check of tag, version, and
  pre-release status ([#302](https://github.com/UniGrazMath/vitabel/pull/302),
  [#304](https://github.com/UniGrazMath/vitabel/pull/304)).
- Locked development dependencies updated with security fixes
  ([#305](https://github.com/UniGrazMath/vitabel/pull/305)).

Contributors: @behackl, @sorlob.

## [0.1.1] - 2026-03-24

`v0.1.1` focuses on smoother data import, more reliable interactive
annotation, expanded respiratory and ventilation analysis, and a more polished
documentation experience.

### Highlights

- Added **EDF import support**.
- Improved data loading robustness, including better handling of **missing
  values** and compatibility with **old and new corpuls nomenclature**.
- Fixed interactive annotation so the current **zoom level is preserved**
  while adding annotations.
- Expanded respiratory and ventilation tooling with:
  - improved **respiratory phase detection**
  - support for filtering by **interval labels**
  - new **inspiratory/expiratory, tidal, and cumulative volume** calculations
  - improved handling of **reverse airflow** and **SI units**
  - clearer plot styling and labeling for respiratory metrics
- Refined the project presentation across the package metadata, README, and
  docs to better frame `vitabel` as an open-source Python framework for
  working with high-resolution physiological time series in critical care
  and perioperative research.
- Improved the documentation experience with a **versioned documentation
  title** and a new **fundamentals notebook** covering `Channel`, `Label`,
  `IntervalLabel`, and `TimeDataCollection`.
- Improved package usability and maintenance with a better `Channel` string
  representation, updated installation/docs/citation metadata, and more
  stable tests after dependency updates.

### Merged pull requests

- EDF integration ([#270](https://github.com/UniGrazMath/vitabel/pull/270), @sorlob)
- Preserve zoom during annotation ([#273](https://github.com/UniGrazMath/vitabel/pull/273), @behackl)
- Citation metadata and README fixes ([#261](https://github.com/UniGrazMath/vitabel/pull/261), @behackl)
- Stabilize tests after dependency bump ([#272](https://github.com/UniGrazMath/vitabel/pull/272), @behackl)
- Add string representation to `Channel` ([#267](https://github.com/UniGrazMath/vitabel/pull/267), @sorlob)
- Add Conda installation instructions to README ([#240](https://github.com/UniGrazMath/vitabel/pull/240), @sorlob)
- Adapt corpuls loading to old and new nomenclature ([#254](https://github.com/UniGrazMath/vitabel/pull/254), @sorlob)
- Improve `na_values` handling during loading ([#252](https://github.com/UniGrazMath/vitabel/pull/252), @sorlob)

## [0.1.0] - 2025-06-30

New contributors: @sorlob, @grahamk1chop, @JohannesUWittig.

### Breaking changes

- Refactor `find_CC_periods` to return `IntervalLabel` ([#164](https://github.com/UniGrazMath/vitabel/pull/164), @sorlob)
- Add auxiliary data structure `DataSlice` holding the output of `Channel.get_data` / `Label.get_data` ([#184](https://github.com/UniGrazMath/vitabel/pull/184), @behackl)

### Added / changed

- Add method for truncating Vitals, Channel, Label ([#21](https://github.com/UniGrazMath/vitabel/pull/21), @behackl)
- Feature: plot `IntervalLabel` as filled background regions when they don't hold y-data ([#28](https://github.com/UniGrazMath/vitabel/pull/28), @behackl)
- Fix typo in demo notebook ([#29](https://github.com/UniGrazMath/vitabel/pull/29), @sorlob)
- Improve initialization of variable in demo notebook ([#30](https://github.com/UniGrazMath/vitabel/pull/30), @sorlob)
- Make override for unit of `ppg` (via `SpO2 %, Waveform`) more robust ([#37](https://github.com/UniGrazMath/vitabel/pull/37), @behackl)
- Update stylesheet.py for Bedmaster at CHOP ([#44](https://github.com/UniGrazMath/vitabel/pull/44), @grahamk1chop)
- Add keyword argument for filtering labels by their type ([#34](https://github.com/UniGrazMath/vitabel/pull/34), @sorlob)
- Use `Rectangle` from `matplotlib` to filter plotting keyword arguments of `IntervalLabel` ([#47](https://github.com/UniGrazMath/vitabel/pull/47), @sorlob)
- fix #71: Add absolute time to data frame generated by `_timeseries_list_info` ([#83](https://github.com/UniGrazMath/vitabel/pull/83), @sorlob)
- Reorder columns in `info` tables ([#90](https://github.com/UniGrazMath/vitabel/pull/90), @sorlob)
- Let overview plots range over full range of plotted data by default; add `limited_overview` keyword argument ([#52](https://github.com/UniGrazMath/vitabel/pull/52), @sorlob)
- Add typing and improve implementation, documentation of datatype argument ([#102](https://github.com/UniGrazMath/vitabel/pull/102), @sorlob)
- Various improvements to CSV / data frame loading code ([#104](https://github.com/UniGrazMath/vitabel/pull/104), @behackl)
- Feature: area under threshold calculations ([#77](https://github.com/UniGrazMath/vitabel/pull/77), @sorlob)
- Add library version to exported data and check for version mismatch when loading ([#106](https://github.com/UniGrazMath/vitabel/pull/106), @behackl)
- Fix and improve handling of time offset, allow directly setting offset attribute ([#111](https://github.com/UniGrazMath/vitabel/pull/111), @behackl)
- Basic support: allow None / pandas.NaT / numpy.nan in time_index ([#113](https://github.com/UniGrazMath/vitabel/pull/113), @behackl)
- Cleanup label string data refactor (continued) ([#133](https://github.com/UniGrazMath/vitabel/pull/133), @behackl)
- Introduce `text_data` for `Label` and `IntervalLabel` ([#131](https://github.com/UniGrazMath/vitabel/pull/131), @sorlob)
- Improve interface of `to_csv` method ([#143](https://github.com/UniGrazMath/vitabel/pull/143), @sorlob)
- Minor improvements to automatic analysis methods ([#135](https://github.com/UniGrazMath/vitabel/pull/135), @sorlob)
- Improve code for choosing automatic preset for plot settings ([#146](https://github.com/UniGrazMath/vitabel/pull/146), @sorlob)
- Refactor dropdown menu ([#140](https://github.com/UniGrazMath/vitabel/pull/140), @sorlob)
- Properly support `text_data` for `IntervalLabel` ([#137](https://github.com/UniGrazMath/vitabel/pull/137), @sorlob)
- Add Label for etco2 ([#165](https://github.com/UniGrazMath/vitabel/pull/165), @sorlob)
- Refactor input to `area_under_threshold` in form of new `source` argument ([#175](https://github.com/UniGrazMath/vitabel/pull/175), @sorlob)
- Add selection for annotation preset for labels ([#177](https://github.com/UniGrazMath/vitabel/pull/177), @sorlob)
- Include adjacent non-included data points when plotting to avoid plots breaking at the ends ([#192](https://github.com/UniGrazMath/vitabel/pull/192), @behackl)
- Add `first_entry` and `last_entry` as properties of `TimeSeriesBase` ([#198](https://github.com/UniGrazMath/vitabel/pull/198), @sorlob)
- Make sure local `vitabel` environment is installed for Binder environments ([#195](https://github.com/UniGrazMath/vitabel/pull/195), @sorlob)
- Add visual feedback on adding partial `IntervalLabel` annotation ([#207](https://github.com/UniGrazMath/vitabel/pull/207), @sorlob)
- Allow creating `IntervalLabel` by list of tuples of interval endpoints ([#202](https://github.com/UniGrazMath/vitabel/pull/202), @sorlob)
- Add `Vitals.detach_label_from_channel` as dedicated way of using data container to manage label attachments ([#178](https://github.com/UniGrazMath/vitabel/pull/178), @behackl)
- Add code used in case processing of German Resuscitation Registry ([#84](https://github.com/UniGrazMath/vitabel/pull/84), @sorlob)
- Allow filtering channels by passing keyword arguments in more methods ([#209](https://github.com/UniGrazMath/vitabel/pull/209), @sorlob)
- feature: Add rename method to `Label` for updating label names ([#212](https://github.com/UniGrazMath/vitabel/pull/212), @sorlob)
- Refactor `Vitals.add_vitaldb_recording` ([#215](https://github.com/UniGrazMath/vitabel/pull/215), @sorlob)
- Improve information displayed in `info` table ([#214](https://github.com/UniGrazMath/vitabel/pull/214), @sorlob)
- Allow querying channels / labels by their index ([#220](https://github.com/UniGrazMath/vitabel/pull/220), @behackl)
- Unify input arguments and checks for automatic analysis / computation methods ([#219](https://github.com/UniGrazMath/vitabel/pull/219), @behackl)
- Improve loading routine for stryker devices to extract metadata ([#221](https://github.com/UniGrazMath/vitabel/pull/221), @sorlob)
- Improve sorting of columns in `info` table ([#224](https://github.com/UniGrazMath/vitabel/pull/224), @sorlob)
- Preparation of demo notebooks ([#144](https://github.com/UniGrazMath/vitabel/pull/144), @sorlob)
- Add method for creating time-scaled time series ([#227](https://github.com/UniGrazMath/vitabel/pull/227), @behackl)
- Add method for importing ventilatory feedback from EOlife export ([#103](https://github.com/UniGrazMath/vitabel/pull/103), @sorlob)

### Fixed

- Fix textual labels on (fixed) axis height ([#25](https://github.com/UniGrazMath/vitabel/pull/25), @behackl)
- Improve robustness of loading ZOLL-json format ([#42](https://github.com/UniGrazMath/vitabel/pull/42), @behackl)
- Use correct method for filtering labels in `Vitals` class ([#61](https://github.com/UniGrazMath/vitabel/pull/61), @sorlob)
- Check if artist was generated before trying to set corresponding label ([#56](https://github.com/UniGrazMath/vitabel/pull/56), @sorlob)
- Cast data time types to avoid `UFuncTypeError` ([#68](https://github.com/UniGrazMath/vitabel/pull/68), @sorlob)
- Make `Vitals.rec_start` and `Vitals.rec_stop` more robust in case of no contained channels ([#74](https://github.com/UniGrazMath/vitabel/pull/74), @sorlob)
- Handle timestamp addition type error ([#80](https://github.com/UniGrazMath/vitabel/pull/80), @sorlob)
- Fix wrong `get_labels` call ([#86](https://github.com/UniGrazMath/vitabel/pull/86), @sorlob)
- Fix wrong `type` call ([#87](https://github.com/UniGrazMath/vitabel/pull/87), @sorlob)
- Improve time type detection in `TimeSeriesBase` class ([#82](https://github.com/UniGrazMath/vitabel/pull/82), @sorlob)
- Check unique timestamps ([#94](https://github.com/UniGrazMath/vitabel/pull/94), @sorlob)
- Improve handling of `time_index` containing `np.NaT` entries ([#109](https://github.com/UniGrazMath/vitabel/pull/109), @sorlob)
- Fix behavior of warning about empty time range query in `get_time_mask` ([#122](https://github.com/UniGrazMath/vitabel/pull/122), @sorlob)
- Fix `IntervalLabel.from_dict` ([#170](https://github.com/UniGrazMath/vitabel/pull/170), @sorlob)
- Set plot legend only once for `axvspan` ([#166](https://github.com/UniGrazMath/vitabel/pull/166), @sorlob)
- Make interactive plot checkbox menu population more robust ([#180](https://github.com/UniGrazMath/vitabel/pull/180), @behackl)
- Fix repainting of overview indicators after changes to overview plot ([#182](https://github.com/UniGrazMath/vitabel/pull/182), @behackl)
- Improve documentation and behavior of methods removing channels or labels ([#183](https://github.com/UniGrazMath/vitabel/pull/183), @behackl)
- Fix: make sure shift marker is redrawn when interactive plot is repainted ([#189](https://github.com/UniGrazMath/vitabel/pull/189), @behackl)
- Allow `Vitals`-import of `pd.DataFrame` with relative time index ([#193](https://github.com/UniGrazMath/vitabel/pull/193), @behackl)
- Checked for empty channels before raising time type inconsistency error ([#200](https://github.com/UniGrazMath/vitabel/pull/200), @sorlob)
- Improve robustness of (interactive) plotting when empty channels / labels are specified ([#204](https://github.com/UniGrazMath/vitabel/pull/204), @sorlob)
- Removed downsampling happening during vitaldb import ([#226](https://github.com/UniGrazMath/vitabel/pull/226), @sorlob)
- Recompute resolution based on current figure width when resizing canvas ([#239](https://github.com/UniGrazMath/vitabel/pull/239), @behackl)

### Documentation

- Setup basic API reference documentation ([#14](https://github.com/UniGrazMath/vitabel/pull/14), @behackl)
- API doc improvements ([#17](https://github.com/UniGrazMath/vitabel/pull/17), @behackl)
- Update README.md ([#40](https://github.com/UniGrazMath/vitabel/pull/40), @sorlob)
- Improve docstrings of `save_data`, `load_data` ([#39](https://github.com/UniGrazMath/vitabel/pull/39), @sorlob)
- Format DOIs as hyperlinks ([#59](https://github.com/UniGrazMath/vitabel/pull/59), @sorlob)
- Updated docstring for `get_label` ([#66](https://github.com/UniGrazMath/vitabel/pull/66), @sorlob)
- Improve documentation by creating various static pages and adding content to existing ones ([#97](https://github.com/UniGrazMath/vitabel/pull/97), @behackl)
- Include demo notebooks as statically rendered subpages in documentation ([#231](https://github.com/UniGrazMath/vitabel/pull/231), @behackl)
- Add proper bibliography in documentation ([#232](https://github.com/UniGrazMath/vitabel/pull/232), @behackl)
- Extend bibliography, add reference to Zotero group ([#237](https://github.com/UniGrazMath/vitabel/pull/237), @sorlob)
- Miscellaneous cleanup in bibliography ([#238](https://github.com/UniGrazMath/vitabel/pull/238), @sorlob)

## [0.0.1] - 2025-01-16

Initial release.

[0.2.0a1]: https://github.com/UniGrazMath/vitabel/compare/v0.1.1...v0.2.0a1
[0.1.1]: https://github.com/UniGrazMath/vitabel/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/UniGrazMath/vitabel/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/UniGrazMath/vitabel/releases/tag/v0.0.1
