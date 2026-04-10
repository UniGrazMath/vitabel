# Quickstart

`vitabel` is an open-source Python framework for working with heterogeneous,
high-resolution physiological time series in retrospective research workflows.
It combines loading, visualization, alignment, processing, and annotation in a
single Jupyter-based environment.

The latest stable release of `vitabel` is distributed via PyPI and can be installed via
```sh
$ pip install vitabel
```

The latest development version can be installed [from the `main` branch on
GitHub](https://github.com/UniGrazMath/vitabel) by running
```sh
$ pip install git+https://github.com/UniGrazMath/vitabel.git
```

See the code example [at the documentation landing page](#/index) for a typical
minimal workflow. More detailed examples, including the required test data, are
contained in the [examples directory](https://github.com/UniGrazMath/vitabel/tree/main/examples).

You can also explore the package directly in your browser via Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UniGrazMath/vitabel/main?urlpath=%2Flab%2Ftree%2Fexamples)


## Use case: German Resuscitation Registry

`vitabel` is used within the German Resuscitation Registry to automatically
extract and review information from defibrillator recordings. For debriefing
and quality improvement, users are provided with an interactive timeline of
resuscitation measures.

Find an example for the interactive timeline in the
embedded page below, or visit the example
[in the registry directly](https://www.reanimationsregister.de/dbshowcase/mgd/public/?PATID=2476802110000BA).

<div class="iframe-container">
<iframe src="https://www.reanimationsregister.de/dbshowcase/mgd/public/?PATID=2476802110000BA" frameborder="0" allowfullscreen scrolling />
</div>

