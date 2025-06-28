<table width="100%">
  <tr>
    <td width="170"><img src="https://raw.githubusercontent.com/UniGrazMath/vitabel/main/assets/logo/Vitabel_Logo.png" width="150"></td>
    <td width="850">
      <b><h1 style="margin: 0;">vitabel</h1></b>
      <p style="margin-top: 0;">a toolbox for interactively annotating and labeling vital data</p>
    </td>
  </tr>
</table>

[![Documentation Status](https://readthedocs.org/projects/vitabel/badge/?version=latest)](https://vitabel.readthedocs.io/en/latest/index.html)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UniGrazMath/vitabel/main?urlpath=%2Flab%2Ftree%2Fexamples)

In a nutshell, the vitabel package enables interactive loading, processing, and annotation of vital medical time-series data (e.g., from defibrillators, anesthesia records, or critical care monitors) within a Jupyter notebook. By structuring and labeling data efficiently and intuitively, it paves the way for AI-driven analysis.

![vitabel annotation screenshot](assets/vitabel-demo.png)

### Interactive Demo in the Browser

We have setup an interactive demo illustrating some standard use cases of `vitabel`
with the help of Binder: [head over to mybinder.org](https://mybinder.org/v2/gh/UniGrazMath/vitabel/main?urlpath=%2Flab%2Ftree%2Fexamples),
or click the binder badge at the top of this README to access the demo right in
your browser. 

### Installation and Usage

The latest stable release of `vitabel` is distributed via PyPI and can be installed via
```sh
$ pip install vitabel
```

The latest development version can be installed [from the `main` branch on
GitHub](https://github.com/UniGrazMath/vitabel) by running
```sh
$ pip install git+https://github.com/UniGrazMath/vitabel.git
```

The main feature of `vitabel`, interactive plots that can be used to annotate data,
is designed to work in Jupyter notebooks. Start a new server by running `jupyter notebook`
(or create a new notebook in an existing server), then import the central `Vitals` class
that acts as a container for the vital data. A set of data can be added using, for example,
the `Vitals.add_defibrillator_recording` method, or `Vitals.add_vital_db_recording`; various output formats of defibrillators
and VitalDB are supported.  

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

More detailed explicit examples (including the required test data) are
contained in the [examples directory](/examples/).

### 📚 Documentation

You can find the full API documentation here: [vitabel.readthedocs.io – vitals module](https://vitabel.readthedocs.io/en/latest/autoapi/vitabel/vitals/index.html)

### 🛠️ Development

Setup a development environment by using the Python project and environment [management
tool `uv`](https://docs.astral.sh/uv/). To setup the environment, simply run
```sh
uv sync
```

Package tests are contained in [the `tests` directory](/tests/); run them locally via
```sh
uv run pytest
```

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting the code base,
and [semantic versioning](https://semver.org/) for the release tags.
