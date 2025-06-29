# vitabel

This package provides a toolbox for interactively annotating
and labeling vital data.

The main feature of `vitabel`, interactive plots that can be used to annotate data,
is designed to work in Jupyter notebooks. Start a new server by running `jupyter notebook`
(or create a new notebook in an existing server), then import the central `Vitals` class
that acts as a container for the vital data.

A set of data can be added using, for example,
the `Vitals.add_defibrillator_recording` method, or `Vitals.add_vital_db_recording`;
various output formats of defibrillators
and VitalDB are supported.  

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

