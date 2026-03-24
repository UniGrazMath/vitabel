# vitabel

`vitabel` is an open-source Python framework for post-hoc loading, visualizing,
aligning, and annotating high-resolution physiological time series.

It is designed for retrospective critical care and perioperative research, where
recordings are often heterogeneous, noisy, and distributed across multiple devices
and file formats. `vitabel` helps turn these data into curated, analysis-ready
datasets by combining interactive visualization, manual annotation, timeline
alignment, and reusable processing workflows in a single Jupyter-based environment.

The framework provides sensible defaults for common use cases while remaining
flexible and extensible for project-specific analysis and annotation pipelines.
Data can be added, for example, via `Vitals.add_defibrillator_recording` or
`Vitals.add_vital_db_recording`; multiple defibrillator formats and VitalDB-based
workflows are supported.

![vitabel annotation demo](_static/img/vitabel-demo.png)

A typical use of this package reads as follows:

```py
from vitabel import Vitals, Label

# create case and load data
case = Vitals()
case.add_defibrillator_recording("path/to/ZOLL_data_file.json")

# use in-built methods for processing available data, compute etco2
# and predict circulatory state
case.compute_etco2_and_ventilations()
case.predict_circulation()

# create a new label for ROSC events
ROSC_label = Label('ROSC', plotstyle={'marker': '$\u2665$', 'color': 'red', 'ms': 10, 'linestyle': ''})
case.add_global_label(ROSC_label)

# display an interactive plot that allows annotations and further data adjustments
case.plot_interactive(
    channels=[['cpr_acceleration'], ['capnography'], ['ecg_pads'], []],
    labels = [['ROSC'], ['etco2_from_capnography', 'ROSC'], ['ROSC'], ['ROSC', 'rosc_probability']],
    channel_overviews=[['cpr_acceleration']],
    time_unit='s',
    subplots_kwargs={'figsize': (22, 9)}
)
```

```{toctree}
:maxdepth: 3
quickstart
examples
development
bibliography
```

