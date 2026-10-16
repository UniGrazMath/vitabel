"""Core module, contains the central data container class :class:`Vitals`."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import logging

from typing import Any, Literal

from IPython.display import display
from pathlib import Path

from vitabel.analysis.ventilation.capnography import detect_etco2_and_ventilations
from vitabel.analysis.ventilation.phases import detect_respiratory_phases
from vitabel.analysis.ventilation.volumes import (
    compute_breath_duration_and_rate_labels,
    compute_cumulative_volume_channels,
    compute_inspiratory_pressure_labels,
    compute_reverse_airflow_labels,
    compute_volume_channels_and_labels,
)
from vitabel.timeseries import (
    DATA_HASH_VERSION,
    Channel,
    Label,
    IntervalLabel,
    TimeDataCollection,
    _timeseries_list_info,
)
from vitabel.analysis.circulation.rosc import predict_rosc
from vitabel.analysis.cpr.defibrillation import extract_shocks
from vitabel.analysis.cpr.compressions import (
    cc_periods_from_acceleration,
    cc_periods_from_compressions,
)
from vitabel.analysis.intervals import filter_by_intervallabel
from vitabel.analysis.threshold import area_under_threshold
from vitabel.utils.serialization import NumpyEncoder
from vitabel.utils import DEFAULT_PLOT_STYLE
from vitabel.io.defibrillator import read_defibrillator_file
from vitabel.io.edf import read_edfplus
from vitabel.io.eolife import read_eolife_export
from vitabel.io.tabular import (
    iter_series_from_dataframe,
    iter_series_from_dict,
    read_csv_table,
    series_from_dict,
)
from vitabel.io.vitaldb import assign_vitaldb_labels, read_vitaldb_tracks
from vitabel.typing import (
    Timedelta,
    Timestamp,
    ChannelSpecification,
    LabelSpecification,
    ThresholdMetrics,
)

logger: logging.Logger = logging.getLogger("vitabel")
"""Global package logger."""


class Vitals:
    """Container for vital data and labels, central interface of this package.

    The Vitals class supports adding data using various methods, such
    as loading data from files directly via :meth:`add_defibrillator_recording`,
    or :meth:`add_vital_db_recording`. It also supports adding data
    channels and labels directly from a pandas ``DataFrame``
    via :meth:`add_data_from_DataFrame`.

    Internally, the data is stored using the :class:`.TimeDataCollection`
    class, which stores data channels and labels as :class:`.Channel`
    and :class:`.Label` objects, respectively. These can also be added
    directly to the Vitals object using :meth:`add_channel` and
    :meth:`add_global_label`.

    Examples
    --------

    ::

        >>> import pandas as pd
        >>> from vitabel import Vitals, Channel
        >>> case = Vitals()
        >>> event_channel = Channel(
        ...     "events",
        ...     pd.date_range("2021-01-01", periods=10, freq="H"),
        ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ... )
        >>> case.add_channel(event_channel)
    """

    def __init__(self):
        self.data: TimeDataCollection = TimeDataCollection()
        """The internal data collection containing all channels and labels."""

        self.metadata: dict = {
            "Time_of_construction": str(
                pd.Timestamp.now()
            ),  # Contains background data, hashes and file names of the object
            "Recording_files_added": [],
            "Saving_to_json_time": "",
        }
        """Metadata dictionary containing information about the data collection."""

        self.start_time = 0  # Start Time for the entire case
        """The start time of the recording."""

    @property
    def channels(self) -> list[Channel]:
        """List of all channels in the data collection."""
        return self.data.channels

    @property
    def labels(self) -> list[Label]:
        """List of all labels in the data collection."""
        return self.data.labels

    def print_data_summary(self) -> None:
        """Print a summary of the data contained in the internal data collection."""
        self.data.print_summary()

    ###
    ### Data import
    ###

    def add_defibrillator_recording(
        self, filepath: Path | str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add the (defibrillator) recording to the cardio object.

        Imports from the following defibrillator device families are supported:

        * **ZOLL X-Series:** data needs to be exported from device as JSON or XML.
        * **ZOLL E-Series and ZOLL AED-Pro:** Data needs to be exported
          from the device in two files, ``filename_ecg.txt`` and ``filename.xml``,
          where ``filename`` can be arbitrary. Both files are assumed to be in the
          same directory. Pass the path to ``filename_ecg.txt`` to import the data.
        * **Stryker LIFEPAK 15:** Data needs to be exported to XML in *Stryker's
          CodeStat Software*. To load the data, we require at least the files
          ``filename_Continuous.xml``, ``filename_Continuous_Waveform.xml``, and
          ``filename_CprEventLog.xml``, where ``filename`` is an arbitrary prefix.
          All files are assumed to be in the same directory. Pass the path
          to ``filename_Continuous.xml`` to import the data.
        * **Stryker LUCAS:** Data needs to be exported to XML in *Stryker's CodeStat
          Software*. We require at leas the files ``filename_Lucas.xml`` and
          ``filename_CprEventLog.xml``, where ``filename`` is an arbitrary prefix.
          Both files are assumed to be in the same directory. Pass the path to
          ``filename_Lucas.xml`` to import the data.
        * **Corpuls:** The data needs to be exported as a *BDF file* in Corpuls
          analysis software. The export will create a ``filename.bdf`` file,
          containing the waveform data, as well as a directory with various other
          files containing event logs. We assume this directory
          is placed in the same directory as the ``.bdf`` file. Pass the path to
          ``filename.bdf`` to import the data.

        The actual loading routines are implemented in the :mod:`.io` subpackage.

        Parameters
        ----------
        filepath
            The path to a file exported from the defibrillator. Check the
            description above to see, depending on the device type, which
            file path should be passed.
        metadata
            Metadata to be added to the imported data. The passed dictionary
            is not modified.
        """
        metadata = dict(metadata) if metadata is not None else {}
        result = read_defibrillator_file(filepath)
        if result is None:
            return None
        pat_dat, dats = result

        if pat_dat:
            for key in ["units", "Serial No", "File Name"]:
                if key in pat_dat["Main data"].index:
                    metadata[key] = pat_dat["Main data"].loc[key].iloc[0]
            if "Model" in pat_dat["Main data"].index:
                metadata["source"] = pat_dat["Main data"].loc["Model"].iloc[0]

        # Convert the data into channels and add to the channel class.
        for channel_name in dats:
            if len(dats[channel_name].index) >= 1:
                if isinstance(dats[channel_name], pd.DataFrame):
                    if len(dats[channel_name].columns) > 1:
                        for col in dats[channel_name].columns:
                            new_channel_name = channel_name + "_" + col
                            chan = Channel(
                                name=new_channel_name,
                                time_index=np.array(dats[channel_name].index),
                                data=np.array(dats[channel_name][col].values),
                                plotstyle=DEFAULT_PLOT_STYLE.get(
                                    new_channel_name, None
                                ),
                                metadata=metadata,
                            )
                            self.data.add_channel(chan)

                    else:
                        chan = Channel(
                            name=channel_name,
                            time_index=np.array(dats[channel_name].index),
                            data=np.array(dats[channel_name][channel_name].values),
                            plotstyle=DEFAULT_PLOT_STYLE.get(channel_name, None),
                            metadata=metadata,
                        )
                        self.data.add_channel(chan)

                elif isinstance(dats[channel_name], pd.Series):
                    chan = Channel(
                        name=channel_name,
                        time_index=np.asarray(dats[channel_name]),
                        data=None,
                        plotstyle=DEFAULT_PLOT_STYLE.get(channel_name, None),
                        metadata=metadata,
                    )
                    self.data.add_channel(chan)

        self.metadata["Recording_files_added"].append(str(filepath))

    def add_ventilatory_feedback(
        self,
        filepath: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add ventilatory feedback from a given data source.

        Currently only EOlife export files (exported CSV files) are supported.

        Parameters
        ----------
        filepath
            The path to the data source file.
        metadata
            Metadata to be added to the imported data. If not provided, an empty
            dictionary is used.
        """
        if metadata is None:
            metadata = {}
        eolife_export = read_eolife_export(filepath)
        self.add_data_from_DataFrame(
            source=eolife_export.data,
            time_start=eolife_export.recording_start,
            datatype="channel",
            metadata=metadata | eolife_export.metadata,
            column_metadata=eolife_export.column_metadata,
        )

    def add_vital_db_recording(
        self,
        filepath: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Loading channels and labels from a vitalDB recording.

        Parameters
        ----------
        filepath
            The path to the recording. Must be a ``*.vit`` file.
        metadata
            Metadata of the imported channels and labels. Defaults to
            ``{"source": "VitalDB-Recording"}``.

        String tracks are imported as labels and attached to the channels
        they belong to, see :func:`.io.vitaldb.assign_vitaldb_labels`.
        """
        channels, labels = read_vitaldb_tracks(filepath, metadata=metadata)
        for channel in channels:
            self.add_channel(channel)

        attachments, global_labels = assign_vitaldb_labels(
            labels, self.get_channel_names()
        )
        for label, channel_names in attachments:
            for channel_name in channel_names:
                for channel in self.get_channels(channel_name):
                    channel.attach_label(label)
        for label in global_labels:
            self.add_global_label(label)

        self.metadata["Recording_files_added"].append(str(filepath))

    def add_edfplus(
        self,
        file_path: Path | str,
        channel_names: list[str] | None = None,
        label_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load channels and labels from an EDF+ file.

        Reads biosignal channels and annotation labels from an EDF+ file
        using :func:`.io.edf.read_edfplus` and adds them to the
        internal data collection.

        Parameters
        ----------
        file_path
            Path to the EDF+ file.
        channel_names
            List of channel names to import. If ``None``, all channels
            are imported.
        label_names
            List of label names to import. If ``None``, all labels
            are imported.
        metadata
            Additional metadata to merge into the recording metadata.
            If ``None``, no extra metadata is added.
        """
        if metadata is None:
            metadata = {}

        channels, labels, recording_metadata = read_edfplus(file_path)

        if channel_names is not None:
            channels = [ch for ch in channels if ch.name in channel_names]

        if label_names is not None:
            labels = [lb for lb in labels if lb.name in label_names]

        for ch in channels:
            self.data.add_channel(ch)

        for lb in labels:
            self.data.add_global_label(lb)

        self.metadata.update(recording_metadata)
        if metadata:
            self.metadata.update(metadata)
        self.metadata["Recording_files_added"].append(str(file_path))

    def add_old_cardio_label(self, filepath: Path | str) -> None:
        """Add labels from legacy version of this package.

        Can read both consensual as well as singular annotations.

        Parameters
        ----------
        filepath
            Path to legacy-style label file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Annotation {filepath} not found.")

        if filepath.suffix == ".json":
            label_dict = {}
            with open(filepath, "r") as fp:  # Load Annotations
                ann_dict = json.load(fp)
            for label in ann_dict["Merged"]:
                if label != "Time":
                    label_dict[label] = {
                        "timestamp": ann_dict["Merged"][label],
                        "data": None,
                    }
            for label in ann_dict["One-Annotator"]:
                if label != "ZOLL CCF":
                    if label in label_dict:
                        label_dict[label]["timestamp"] = np.append(
                            label_dict[label]["timestamp"],
                            ann_dict["One-Annotator"][label],
                        )
                    else:
                        label_dict[label] = {
                            "timestamp": ann_dict["One-Annotator"][label],
                            "data": None,
                        }
            self.add_data_from_dict(
                label_dict, metadata={"Creator": "Consensus"}, datatype="label"
            )
            compression_dict = {}
            for label in ann_dict["Compressions"]:
                if label != "Orignal Compression" and label != "Original Compression":
                    for annotator in ann_dict["Compressions"][label]:
                        if annotator not in compression_dict:
                            compression_dict[annotator] = {}
                        compression_dict[annotator][label] = {
                            "timestamp": ann_dict["Compressions"][label][annotator],
                            "data": None,
                        }

            for annotator in compression_dict:
                if "cc" in self.data.channel_names:
                    compression_channel = self.data.get_channel("cc")
                elif "cc_depth" in self.data.channel_names:
                    compression_channel = self.data.get_channel("cc_depth")
                else:
                    compression_channel = None
                self.add_data_from_dict(
                    compression_dict[annotator],
                    metadata={"Creator": annotator},
                    datatype="label",
                    anchored_channel=compression_channel,
                )

            problematic_dict = {}
            for label in ann_dict["Problematic"]:
                for annotator in ann_dict["Problematic"][label]:
                    if annotator not in compression_dict:
                        problematic_dict[annotator] = {}
                    problematic_dict[annotator][label] = {
                        "timestamp": ann_dict["Compressions"][label][annotator],
                        "data": None,
                    }

            for annotator in problematic_dict:
                self.add_data_from_dict(
                    problematic_dict[annotator],
                    metadata={"Creator": annotator},
                    datatype="label",
                )
        elif filepath.suffix == ".csv":
            anno = pd.read_csv(filepath)
            label_dict = {}
            interval_label_dict = {}
            metadata = {}

            for annot in anno["Type"].unique():
                if annot not in ["Case", "Annotator", "Time", "Duration /s"]:
                    if annot == "Blood gas analysis":
                        interval_label_dict[annot] = {
                            "timestamp": np.array(anno["Value"][anno["Type"] == annot]),
                            "data": None,
                        }
                    else:
                        label_dict[annot] = {
                            "timestamp": np.array(anno["Value"][anno["Type"] == annot]),
                            "data": None,
                        }
                else:
                    metadata[annot] = anno["Value"][anno["Type"] == annot].values[0]

            self.add_data_from_dict(label_dict, datatype="label", metadata=metadata)
            self.add_data_from_dict(
                interval_label_dict, datatype="interval_label", metadata=metadata
            )
        else:
            raise ValueError(
                "The given file {file_path} is not a valid cardio 1.x annotation."
            )

    def _add_single_dict(
        self,
        source: dict[str, Any],
        name: str,
        time_start=None,
        datatype: Literal["channel", "label", "interval_label"] = "channel",
        anchored_channel: Channel | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Adds a channel or label from a dict containing a single timeseries.

        The dict needs to have two keys: 'timestamp' and 'data'.

        Parameters
        ----------
        source
            Contains the data in the from ``{'timestamp': [], 'data' : []}``
        name
            The name of the channel.
        time_start
            time_start value for the timeseries, in case of a relative timeseries. The default is None.
        datatype
            Either ``'channel'`` or ``'label'`` or ``'interval_label'``, depending
            on which kind of container to construct. The default is ``'channel'``.
        anchored_channel
            If a label is constructed, then this is the channel to which the label
            is anchored. Irrelevant if ``datatype`` is ``'channel'``.
        metadata
            Metadata for the timeseries.

        Raises
        ------
        ValueError
            In case the dictionary does not contain keys 'timestamp' and 'data'.

        Returns
        -------
        None.

        """
        series = series_from_dict(
            source,
            name,
            time_start=time_start,
            datatype=datatype,
            anchored_channel=anchored_channel,
            metadata=metadata,
        )
        self._add_imported_series(series)

    def _add_imported_series(self, series: Channel | Label | None) -> None:
        """Add an imported channel, or an imported label that is not attached
        to a channel (as global label), to the data collection."""
        if series is None:
            return
        if isinstance(series, Channel):
            self.data.add_channel(series)
        elif series.anchored_channel is None:
            self.data.add_global_label(series)

    def add_data_from_dict(
        self,
        source: dict[str, dict] | Path,
        time_start=None,
        datatype: Literal["channel", "label", "interval_label"] = "channel",
        anchored_channel: Channel | None = None,
        metadata: dict | None = None,
    ):
        """Add multiple channels from a dictionary.

        Each value must be a dictionary accepted by
        :meth:`._add_single_dict`.

        Parameters
        ----------
        source
            The data which is added in the from
            ``{'key1': {'timestamp': [], 'data': []}, 'key2': { ... }, ... }``.
            If source is a ``Path``, then it is assumed to be a JSON file and loaded
            via ``json.load``.
        time_start
            time_start value for the timeseries, in case of a relative timeseries.
            The default is ``None``.
        datatype
            Either ``'channel'`` or ``'label'`` or ``'interval_label'``, depending
            on which kind of container to construct. The default is ``'channel'``.
        anchored_channel
            If a label is constructed, then this is the channel to which the label
            is anchored. Irrelevant if ``datatype`` is ``'channel'``.
        metadata
            Metadata that will be attached to all individual timeseries.

        Raises
        ------
        ValueError
            In case the dictionary does not have the expected form.
        """
        for series in iter_series_from_dict(
            source,
            time_start=time_start,
            datatype=datatype,
            anchored_channel=anchored_channel,
            metadata=metadata,
        ):
            self._add_imported_series(series)

    def add_data_from_DataFrame(
        self,
        source: pd.DataFrame,
        time_start: str | None = None,
        time_unit=None,
        datatype: Literal["channel", "label", "interval_label"] = "channel",
        anchored_channel: Channel | None = None,
        metadata: dict[str, Any] | None = None,
        column_metadata: dict[str, dict[str, Any]] | None = None,
    ):
        """Adds Data from a ``pandas.DataFrame``.

        Parameters
        ----------
        source
            The DataFrame containing the data. The index of the DataFrame contains the
            time (either as DatetimeIndex, TimedeltaIndex or numeric Index),
            and the columns contain the channels. NaN-Values in the columns are
            not taken into account an ignored.
        time_start
            A starting time for the data. Must be accepted by pd.Timestamp(time_start)
            In case the index is timedelta or numeric. The times will be interpreted as relative
            to this value. The default is 0 and means no information is given.
        time_unit
            The time unit of the data. Must be accepted by pd.Timestamp(time_unit).
        datatype
            Either ``'channel'`` or ``'label'`` or ``'interval_label'``, depending
            on which kind of container to construct. The default is ``'channel'``.
        anchored_channel
            If a label is constructed, then this is the channel to which the label
            is anchored. Irrelevant if ``datatype`` is ``'channel'``.
        metadata
            A dictionary containing all the metadata for the channels / labels.
        column_metadata
            A dictionary containing metadata for individual columns in the
            dataframe. The keys are the column names, the values are dictionaries.
            When storing the data, the metadata from this dictionary is merged
            with the global metadata (and overriding it in case of conflicts).

        Raises
        ------
        ValueError
            The DataFrame does not contain a DateTime or Numeric Index.
        """
        for series in iter_series_from_dataframe(
            source,
            time_start=time_start,
            time_unit=time_unit,
            datatype=datatype,
            anchored_channel=anchored_channel,
            metadata=metadata,
            column_metadata=column_metadata,
        ):
            self._add_imported_series(series)

    def add_data_from_csv(
        self,
        file_path: Path | str,
        time_start=None,
        time_unit=None,
        datatype: Literal["channel", "label", "interval_label"] = "channel",
        anchored_channel: Channel | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Adds data from a CSV file.

        The CSV file must contain a header with the channel names and a
        timestamp column. The data is loaded into a pandas DataFrame and
        passed to the :meth:`add_data_from_DataFrame` method.

        Parameters
        ----------
        file_path
            The path to the CSV file. The file must contain a header with the channel names
            and a timestamp column.
        time_start
            A starting time for the data. Must be accepted by pd.Timestamp(time_start)
            In case the index is numeric. The times will be interpreted as relative
            to this value. The default is 0 and means no information is given.
        time_unit
            The time unit of the data. Must be accepted by pd.Timestamp(time_unit).
        datatype
            Either ``'channel'`` or ``'label'`` or ``'interval_label'``, depending
            on which kind of container to construct. The default is ``'channel'``.
        anchored_channel
            If a label is constructed, then this is the channel to which the label
            is anchored. Irrelevant if ``datatype`` is ``'channel'``.
        metadata
            A dictionary containing all the metadata for the channels/labels.
            Is parsed to channel/Label and saved there as general argument.
        kwargs
            Additional keyword arguments are passed to the ``pandas.read_csv``
            function.
        """
        df = read_csv_table(file_path, **kwargs)
        self.add_data_from_DataFrame(
            df,
            time_start=time_start,
            time_unit=time_unit,
            datatype=datatype,
            anchored_channel=anchored_channel,
            metadata=metadata,
        )

    def add_channel(self, channel: Channel):
        self.data.add_channel(channel)

    def add_global_label(self, label: Label):
        self.data.add_global_label(label)

    # # Add loading iterable of Frame or Series to add more channels at once
    # time_unit = time_unit,
    def remove_channel(self, *, channel: Channel | None = None, **kwargs) -> Channel:
        """Remove a channel from the data collection.

        Parameters
        ----------
        channel
            The channel object to remove. If not provided, the channel is
            fetched from the collection using the other provided
            keyword arguments.
        kwargs
            Keyword arguments to filter the channels by. Passed
            to :meth:`.TimeDataCollection.get_channel`, resulting
            channel needs to be uniquely identified.

        Returns
        -------
        Channel
            The removed channel object.
        """
        return self.data.remove_channel(channel=channel, **kwargs)

    def remove_label(self, *, label: Label | None = None, **kwargs) -> Label:
        """Remove a label from the data collection.

        Parameters
        ----------
        label
            The label object to remove. If not provided, the label is
            fetched from the collection using the other provided
            keyword arguments.
        kwargs
            Keyword arguments to filter the channels by. Passed
            to :meth:`.TimeDataCollection.get_label`, resulting
            label needs to be uniquely identified.

        Returns
        -------
        Label
            The removed label object.
        """
        return self.data.remove_label(label=label, **kwargs)

    def set_channel_plotstyle(
        self, channel_specification: ChannelSpecification | None = None, **kwargs
    ):
        """Set the plot style for a channel.

        Parameters
        ----------
        channel_specification
            A specification of all channels to set the plot style for.
            See :meth:`.get_channels` for valid specifications.
        kwargs
            The plot style properties to set. Passing ``None``
            unsets the key from the plotstyle dictionary.
        """
        self.data.set_channel_plotstyle(channel_specification, **kwargs)

    def set_label_plotstyle(
        self, label_specification: LabelSpecification | None = None, **kwargs
    ):
        """Set the plot style for specified labels.

        Parameters
        ----------
        label_specification
            A specification of all labels to set the plot style for.
            See :meth:`.get_labels` for valid specifications.
        kwargs
            The plot style properties to set. Passing ``None``
            unsets the key from the plotstyle dictionary.
        """
        self.data.set_label_plotstyle(label_specification, **kwargs)

    def save_data(self, path: Path | str):
        """Exports channels, labels, and metadata and saves it in a JSON file.
        In particular, adds a data hash and the vitabel version to the metadata.

        Parameters
        ----------
        path
            The path of the output JSON file.
        """
        from vitabel import __version__

        if isinstance(path, str):
            path = Path(path)

        self.metadata["Saving_to_json_time"] = str(pd.Timestamp.now())
        self.metadata["filepath"] = str(path)
        self.metadata["vitabel version"] = __version__
        data_hash = self.data.channel_data_hash()
        data_dict = self.data.to_dict()
        json_dict = {
            "metadata": self.metadata,
            "data": data_dict,
            "hash": data_hash,
            "hash_version": DATA_HASH_VERSION,
        }

        with open(path, "w") as fd:
            json.dump(json_dict, fd, cls=NumpyEncoder)

    def load_data(self, path: Path | str, check_channel_hash: bool = True):
        """
        Loads all Channels and Labels from a previously saved JSON file.

        Parameters
        ----------
        path
            The Path of the channel data.
        check_channel_hash
            Whether to compare the hash of the loaded channel data with the
            hash stored in the file (see :meth:`save_data`). A mismatch, which
            indicates that the data has been modified outside of vitabel,
            results in a warning. Files without a hash of the current format
            (:data:`.timeseries.DATA_HASH_VERSION`) are not checked. Defaults
            to ``True``; pass ``False`` to skip the (potentially expensive)
            check.
        """
        from vitabel import __version__

        path = Path(path)
        with open(path, "r") as fd:
            json_dict = json.load(fd)
        self.metadata = json_dict["metadata"]
        self.data = TimeDataCollection.from_dict(json_dict["data"])
        vitabel_version = json_dict["metadata"].get("vitabel version", "unknown")
        if vitabel_version != "unknown" and __version__ != vitabel_version:
            logger.warning(
                f"The vitabel version used to save the data is {vitabel_version}, "
                f"while the currently installed version is {__version__}. "
                "This may lead to compatibility issues."
            )

        # files saved with older versions contain no (comparable) hash
        saved_hash = json_dict.get("hash")
        hash_version = json_dict.get("hash_version")
        if (
            check_channel_hash
            and saved_hash is not None
            and hash_version == DATA_HASH_VERSION
        ):
            channel_hash = self.data.channel_data_hash()
            if channel_hash != saved_hash:
                version_note = (
                    f" The file was saved with vitabel {vitabel_version}."
                    if vitabel_version not in ("unknown", __version__)
                    else ""
                )
                logger.warning(
                    f"The channel data loaded from {path} does not match the hash "
                    "stored in the file, which indicates that the data has been "
                    f"modified outside of vitabel.{version_note} Pass "
                    "check_channel_hash=False to skip this check."
                )

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Retrieve data -------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def get_channels(self, name: str | None = None, **kwargs) -> list[Channel]:
        """Returns a list of channels based on their name.

        This method retrieves all channels that match the given name and
        keyword arguments. If no name is specified, all channels are returned.

        Parameters
        ----------
        name
            The name of the channels to retrieve.
        kwargs
            Keyword arguments to filter the channels by.

        Returns
        -------
        list[Channel]
            A list of channels that match the specification.
            If no channel matches the specification, an empty list is returned.
        """
        return self.data.get_channels(name, **kwargs)

    def get_channel(self, name: str | int | None = None, **kwargs) -> Channel:
        """Returns a single channel based on its name.

        Parameters
        ----------
        name
            The name of the channel to retrieve. If an integer is passed,
            it is interpreted as the index of the channel in the list of
            all channels.
        kwargs
            Keyword arguments to filter the channels by.

        Raises
        ------
        ValueError
            If the specification is ambiguous, i.e., if more than one or
            no channel matches the specification.

        Returns
        -------
        Channel
            The channel that matches the specification.
        """

        return self.data.get_channel(name, **kwargs)

    def get_labels(
        self, name: str | None = None, label_type: type[Label] | None = None, **kwargs
    ) -> list[Label]:
        """Returns a list of labels based on their name.

        Parameters
        ----------
        name
            The name of the labels to retrieve.
        label_type : TYPE, optional
            A specification of the label type (IntervalLabel or Label) to retrieve
        kwargs
            Keyword arguments to filter the labels by.

        Returns
        -------
        list[Label]
            A list of labels that match the specification.
            If no label matches the specification, an empty list is returned.
        """

        if label_type is not None:
            return [
                label for label in self.get_labels(name) if type(label) is label_type
            ]
        return self.data.get_labels(name, **kwargs)

    def get_label(self, name: str | int | None = None, **kwargs) -> Label:
        """Return a list of labels.

        Parameters
        ----------
        name
            The name of the label to retrieve. Allowed to be passed
            either as a positional or a keyword argument. If ``name`` is an integer,
            it is interpreted as the index of the label in the list of all labels.
        kwargs
            Keyword arguments to filter the labels by. The
            specified arguments are compared to the attributes
            of the labels.
        """
        return self.data.get_label(name, **kwargs)

    def get_channels_or_labels(
        self, name: str | None = None, label_type: type[Label] | None = None, **kwargs
    ) -> list[Channel | Label]:
        """Returns a list of channels or labels based on their name.

        This method combines the results of :meth:`.get_channels`
        and :meth:`.get_labels` to return both channels and labels
        that match the given parameters.
        If a ``label_type`` is specified, it filters the labels accordingly.


        Parameters
        ----------
        name
            The name of the channels or labels to retrieve.
            See :meth:`.get_channels` and :meth:`.get_labels` for valid specifications.
        label_type
            The class of the labels (e.g., :class:`.IntervalLabel`) to
            retrieve.
        kwargs
            Keyword arguments to filter the channels or labels by.
            See :meth:`.get_channels` and :meth:`.get_labels` for valid specifications.

        Returns
        -------
        list[Channel | Label]
            A list of channels or labels that match the specification.
            If no channel or label matches the specification, an empty list is returned.

        """
        return self.data.get_channels(name, **kwargs) + self.get_labels(
            name, label_type=label_type, **kwargs
        )

    def get_channel_or_label(
        self, name: str | None = None, label_type: type[Label] | None = None, **kwargs
    ) -> Channel | Label:
        """Returns a single channel or label based on its name.

        Parameters
        ----------
        name
            The name of the channel or label to retrieve.
            See :meth:`.get_channel` and :meth:`.get_label` for valid specifications.
        label_type
            The class of the label (e.g., :class:`.IntervalLabel`) to
            retrieve.
        kwargs
            Keyword arguments to filter the channels or labels by.
            See :meth:`.get_channel` and :meth:`.get_label` for valid specifications.

        Raises
        ------
        ValueError
            If the specification is ambiguous, i.e., if more than one channel or label
            matches the specification.

        Returns
        -------
        Channel | Label
            The channel or label that matches the specification.
        """

        channels_or_labels = self.get_channels_or_labels(
            name, label_type=label_type, **kwargs
        )
        if len(channels_or_labels) != 1:
            raise ValueError(
                "Channel or Label specification was ambiguous, no unique channel or Label "
                f"was identified. Query returned: {channels_or_labels}"
            )
        return channels_or_labels[0]

    def get_channel_names(self, **kwargs) -> list[str]:  # part of register application
        """Return a list with the names of all channels in the recording.

        Parameters
        ----------
        kwargs
            Keyword arguments to filter the channels by.

        Returns
        -------
        list[str]
            A list with the names of all channels in the recording.
            If no channel matches the specification, an empty list is returned.
        """

        channels = self.data.get_channels(**kwargs)

        return [channel.name for channel in channels]

    def get_label_names(self, **kwargs) -> list[str]:  # part of register application
        """Returns a list with the names of all labels.

        Parameters
        ----------
        kwargs
            Keyword arguments to filter the labels by.
            See :meth:`.get_label` for valid specifications.

        Returns
        -------
        list[str]
            A list with the names of all labels in the recording.
            If no label matches the specification, an empty list is returned.

        """

        labels = self.data.get_labels(**kwargs)

        return [label.name for label in labels]

    def get_channel_or_label_names(self, **kwargs) -> list[str]:
        """Returns a list with all names from either channels or labels.

        Parameters
        ----------
        kwargs
            Keyword arguments to filter the channels and labels by.
            See :meth:`.get_channel` and :meth:`.get_label` for valid specifications.

        Returns
        -------
        list[str]
            A list with the names of all channels and labels in the recording.
            If no channel or label matches the specification, an empty list is returned.
        """
        return self.get_channel_names(**kwargs) + self.get_label_names(**kwargs)

    def keys(self, **kwargs) -> list[str]:
        """Alias for :meth:`get_channel_or_label_names`."""

        return self.get_channel_or_label_names(**kwargs)

    def detach_label_from_channel(
        self,
        *,
        label: Label | str,
        channel: Channel | str | None = None,
        reattach_as_global: bool = True,
    ) -> Label:
        """Detach a label from a channel in the collection.

        Parameters
        ----------
        label
            The label to detach. Can be specified either as a
            :class:`.Label` object or by its name.
        channel
            The channel to detach the label from or ``None`` (the default)
            if the channel should be determined from the label.
            Can be specified either as a :class:`.Channel` object or
            by its name.
        reattach_as_global
            If ``True``, the label is reattached as a global label
            after detaching it from the channel. If ``False``, the
            label is removed from the collection.
        """
        return self.data.detach_label_from_channel(
            label=label, channel=channel, reattach_as_global=reattach_as_global
        )

    def rec_start(self) -> pd.Timestamp:  # part of register application
        """Returns the first timestamp among all channels in this case."""
        if self.data.is_time_absolute():
            start_time = self.data.channels[0].time_start
            for chan in self.channels:
                if chan.time_start < start_time:
                    start_time = chan.time_start

        return start_time

    def rec_stop(self) -> pd.Timestamp:  # part of register application
        """Returns the last timestamp among all channels in this case."""
        if self.data.is_time_absolute():
            stop_time = (
                self.data.channels[0].time_start + self.data.channels[0].time_index[-1]
            )
            for chan in self.channels:
                cha_stop_time = chan.time_start + chan.time_index[-1]
                if cha_stop_time > stop_time:
                    stop_time = cha_stop_time

        return stop_time

    def get_channel_infos(self, **kwargs) -> pd.DataFrame:
        """Returns information about all channels in a nicely formatted
        ``pandas.DataFrame``.
        This includes the channel name, number of datapoints, first entry, last entry, offset, auxiliary metadata.

        Parameters
        ----------
        kwargs
            Keyword arguments to filter the channels by.
            See :meth:`.get_channels` for valid specifications.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the channel information.
        The DataFrame contains the following columns:
        - ``Name``: The name of the channel.
        - ``Length``: The number of data points in the channel.
        - ``First Entry``: The first timestamp of the channel.
        - ``Last Entry``: The last timestamp of the channel.
        - ``Offset``: The offset of the channel, i.e., the time difference
        - ``metadata``: All auxillary metadata of the channel.
        """
        return _timeseries_list_info(self.get_channels(**kwargs))

    def get_label_infos(self, **kwargs) -> pd.DataFrame:
        """Returns information about all labels in a nicely formatted
        ``pandas.DataFrame``.
        This includes the label name, number of datapoints, first entry, last entry, offset, auxiliary metadata.

        Parameters
        ----------
        kwargs
            Keyword arguments to filter the labels by.
            See :meth:`.get_labels` for valid specifications.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the label information.
        The DataFrame contains the following columns:
        - ``Name``: The name of the label.
        - ``Length``: The number of data points in the label.
        - ``First Entry``: The first timestamp of the label.
        - ``Last Entry``: The last timestamp of the label.
        - ``Offset``: The offset of the label, i.e., the time difference
        - ``metadata``: All auxillary metadata of the label.
        """
        return _timeseries_list_info(self.get_labels(**kwargs))

    def info(self, **kwargs) -> None:
        """Display relevant information about the channels and labels
        currently present in the recording in two separate
        ``pandas.DataFrame`` objects.

        Parameters
        ----------
        kwargs
            Keyword arguments to filter the channels and labels by.
            See :meth:`.get_channels` and :meth:`.get_labels` for valid
            specifications.

        Returns
        -------
        None
        """
        channel_info = self.get_channel_infos(**kwargs)
        label_info = self.get_label_infos(**kwargs)
        display(channel_info)
        display(label_info)

    def truncate(
        self,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
        **kwargs,
    ) -> Vitals:
        """Return a new object where time data has been truncated to a specified interval.

        Parameters
        ----------
        start_time
            The start time of the truncated recording.
        stop_time
            The stop time of the truncated recording.
        kwargs
            Additional keyword arguments to filter the channels by

        Returns
        -------
        Vitals
            A new Vitals object containing the truncated recording.
        """
        truncated_vitals = Vitals()
        for channel in self.get_channels(**kwargs):
            truncated_vitals.add_channel(channel.truncate(start_time, stop_time))
        for label in self.data.global_labels:
            truncated_vitals.add_global_label(label.truncate(start_time, stop_time))
        return truncated_vitals

    ###
    ### Plotting
    ###

    def plot(
        self,
        channels: list[list[ChannelSpecification | int]] | None = None,
        labels: list[list[LabelSpecification | int]] | None = None,
        start: Timestamp | Timedelta | float | str | None = None,
        stop: Timestamp | Timedelta | float | str | None = None,
        resolution: Timedelta | float | None = None,
        time_unit: str | None = None,
        include_attached_labels: bool = False,
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        """Plot the data in the collection.

        .. SEEALSO::

            :meth:`.TimeDataCollection.plot`

        Parameters
        ----------
        channels
            The channels to plot. If not specified, all channels are plotted.
            Specified as a list of lists with individual lists containing
            channels to be collected in one subplot.
        labels
            The labels to plot. If not specified, all labels are plotted.
            Specified as a list of lists, same as for the channels.
        start
            The start time for the plot. If not specified, the plot starts
            from the first time point.
        stop
            The stop time for the plot. If not specified, the plot stops
            at the last time point.
        resolution
            The resolution of the plot in the time unit of the channels.
            If not specified, the channel and label data is not downsampled.
        time_unit
            The time unit in which channel and label data are represented
            in. If not specified, the time unit of the channels is used.
        include_attached_labels
            Whether to automatically include labels attached to the
            specified channels.
        subplots_kwargs
            Keyword arguments passed to ``matplotlib.pyplot.subplots``.
        """
        return self.data.plot(
            channels=channels,
            labels=labels,
            start=start,
            stop=stop,
            resolution=resolution,
            time_unit=time_unit,
            include_attached_labels=include_attached_labels,
            subplots_kwargs=subplots_kwargs,
        )

    def plot_interactive(
        self,
        channels: list[list[ChannelSpecification | int]] | None = None,
        labels: list[list[LabelSpecification | int]] | None = None,
        start: Timestamp | Timedelta | float | str | None = None,
        stop: Timestamp | Timedelta | float | str | None = None,
        time_unit: str | None = None,
        include_attached_labels: bool = False,
        channel_overviews: list[list[ChannelSpecification | int]] | bool = False,
        limited_overview: bool = False,
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        """Plot the data in the collection using ipywidgets.

        This allows to annotate the data with labels, and to modify
        channel offsets interactively.

        .. SEEALSO::

            :meth:`.TimeDataCollection.plot_interactive`

        Parameters
        ----------
        channels
            The channels to plot. If not specified, all channels are plotted.
            Specified as a list of lists with individual lists containing
            channels to be collected in one subplot.
        labels
            The labels to plot. If not specified, all labels are plotted.
            Specified as a list of lists, same as for the channels.
        start
            The start time for the plot. If not specified, the plot starts
            from the first time point.
        stop
            The stop time for the plot. If not specified, the plot stops
            at the last time point.
        time_unit
            The time unit in which channel and label data are represented
            in. If not specified, the time unit of the channels is used.
        include_attached_labels
            Whether to automatically include labels attached to the
            specified channels.
        channel_overviews
            Similar to ``channel``, but plots the specified channels
            in a separate subplot in a condensed way including a
            location map of the main plot. If set to ``True``, all
            chosen channels are plotted in a single overview.
        limited_overview
            Whether the time interval of the overview subplots should be limited
            to the recording interval of the channels being plotted.
        subplots_kwargs
            Keyword arguments passed to ``matplotlib.pyplot.subplots``.

        Returns
        -------
        InteractivePlot
            An ipywidgets layout (control panel above the figure) that can be
            displayed and combined with other widgets. It also provides:

            - ``on(event, callback)``: react to changes, e.g.
              ``"labels_changed"`` after annotating;
            - ``add_overlay(overlay, subplot=0)``: persistent overlays, e.g.
              :class:`~vitabel.plotting.overlays.FillBetween` to shade the
              area of a signal below a threshold;
            - ``set_window(start, stop)``;
            - ``figure``: the matplotlib figure (customisations persist when
              the plot is updated).
        """
        return self.data.plot_interactive(
            channels=channels,
            labels=labels,
            start=start,
            stop=stop,
            time_unit=time_unit,
            include_attached_labels=include_attached_labels,
            channel_overviews=channel_overviews,
            limited_overview=limited_overview,
            subplots_kwargs=subplots_kwargs,
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------- AUTOMATIC LABELING --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------

    def shocks(self):
        """Return all defibrillations with all stored information in a
        DataFrame, see :func:`.analysis.cpr.defibrillation.extract_shocks`.

        Returns
        -------
        shocks : pandas.DataFrame
            One row per defibrillation (index: time of the defibrillation).
        """
        return extract_shocks(self.channels)

    def compute_etco2_and_ventilations(
        self,
        source: Channel | str = "capnography",
        mode: Literal["threshold", "filter"] = "filter",
        breath_threshold: float = 2,
        etco2_threshold: float = 3,
        **kwargs,
    ):
        """Computes end-tidal CO2 (etCO₂) values and timestamps of ventilations from the
        capnography waveform, and adds them as labels (attached to the capnography
        channel). See :func:`.analysis.ventilation.capnography.detect_etco2_and_ventilations`.

        The capnography signal must be present as a channel named 'capnography'. Two
        detection methods are supported:

        - ``'filter'``: An unpublished method by Wolfgang Kern (default).
        - ``'threshold'``: The method described by Aramendi et al. in
          :cite:`10.1016/j.resuscitation.2016.08.033`.

        Parameters
        ----------
        mode
            Method to use for detecting ventilations from the CO₂ signal.
        breath_threshold
            Threshold below which a minimum is identified as a ventilation (default: 2 mmHg).
            Used by the ``'filter'`` method.
        etco2_threshold
            Threshold above which a maximum is identified as an etCO₂ value of an expiration
            (default: 3 mmHg). Used by the ``'filter'`` method.
        """
        # Support legacy parameter name
        if "breaththresh" in kwargs:
            if breath_threshold is not None:
                raise TypeError(
                    "Cannot specify both 'breath_threshold' and legacy 'breaththresh'"
                )
            breath_threshold = kwargs.pop("breaththresh")
            logger.warning(
                "The keyword argument breaththresh is deprecated, "
                "use breath_threshold instead"
            )

        if isinstance(source, str):
            source = self.get_channel(name=source)

        if not isinstance(source, Channel):
            raise ValueError(
                f"No valid capnography channel specified. Please make sure a channel named '{source}' "
                "is added to the collection, or specify a suitable channel via "
                "the source argument directly or by name"
            )

        labels = detect_etco2_and_ventilations(
            source,
            mode=mode,
            breath_threshold=breath_threshold,
            etco2_threshold=etco2_threshold,
        )
        if labels is not None:
            for label in labels:
                source.attach_label(label)

    def cycle_duration_analysis(
        self, cc_events_channel: Channel | str | None = None
    ) -> None:
        """
        Determines periods of continuous chest compressions
        based on single chest compression events.

        Parameters
        ----------
        cc_events_channel
            The channel containing the data of single chest compression events, such that every timepoint in the timeindex represents a chest compression.
            Defaults to 'cc' or 'cc_depth' depending on the availability.

        Returns
        -------
        None

        Attaches an IntervalLabel withe the name ``cc_periods`` to the channel of single chest compressions.

        .. SEEALSO::
            The method is described in :cite:`10.1016/j.resuscitation.2021.12.028`, or in the
            PhD thesis :cite:`kern:phd:2024` of Wolfgang Kern in more detail.
        """
        if cc_events_channel is None:
            available_channels = set(self.get_channel_names()) & {"cc", "cc_depth"}
            if available_channels:
                cc_events_channel = self.get_channel(next(iter(available_channels)))
            else:
                raise ValueError(
                    "Could not identify channels with single chest compressions. "
                    "Please specify a channel or a string with the name of the channel."
                )
        elif isinstance(cc_events_channel, str):
            cc_events_channel = self.get_channel(cc_events_channel)
        elif not isinstance(cc_events_channel, Channel):
            raise ValueError(
                "No valid channel with chest compression specified. Can not identify CC-periods via single CCs."
                "Please specify a channel or a string with the name of the channel."
            )

        cc_periods = cc_periods_from_compressions(cc_events_channel)
        cc_events_channel.attach_label(cc_periods)

    def find_CC_periods_acc(
        self,
        accelerometer_channel: Channel | str = "cpr_acceleration",
    ) -> None:  # part of register application
        """
        Automatically detects periods of continuous chest compressions.

        The procedure is implemented as described in :cite:`10.1016/j.resuscitation.2021.12.028`
        and :cite:`10.1016/j.dib.2022.107973`. In essence it uses the root mean
        square of the accelerometer signal of feedback sensor for cardiopulmonary
        resuscitation to detect the rise in "power" of the signal linked to the
        alteration by the accelerations of continous chest compressions.

        Parameters
        ----------
        accelerometer_channel
            The channel containing the accelerometer signal. Defaults to 'cpr_acceleration'.

        Returns
        -------
        None.

        Attaches an IntervalLabel withe the name ``cc_periods`` to the accelerometer channel.
        Every entry in the label describes a single period of chest compressions.

        .. SEEALSO::
            Details are given in :cite:`10.1016/j.resuscitation.2021.12.028` or in the
            PhD thesis :cite:`kern:phd:2024` by Wolfgang Kern in more detail.
        """
        if isinstance(accelerometer_channel, str):
            ACC_channel = self.get_channel(accelerometer_channel)
        elif not isinstance(accelerometer_channel, Channel):
            raise ValueError(
                "No valid accelerometer channel specified. Can not identify CC-periods via acceleration."
                "Please specify a channel or a string with the name of the channel."
            )
        else:
            ACC_channel = accelerometer_channel

        cc_periods = cc_periods_from_acceleration(ACC_channel)
        ACC_channel.attach_label(cc_periods)

    def predict_circulation(
        self,
        cpr_acceleration_source: Channel | str = "cpr_acceleration",
        ecg_pads_source: Channel | str = "ecg_pads",
    ) -> None:
        """Predicts the circulation of a case by using the channels
        ``'cpr_acceleration'`` channel and the ``'ecg_pads'`` channel.

        The procedure that is used has been published by Kern et al.
        in :cite:`10.1109/tbme.2023.3242717`.
        Here ``'rosc_decision_function'`` is the output of the kernelized
        SVM used in and trained for the paper.

        Parameters
        ----------
        cpr_acceleration_source
            The (name of the) channel containing the accelerometer signal of the CPR
            feedback sensor. Defaults to ``'cpr_acceleration'``.
        ecg_pads_source
            The (name of the) channel containing the ECG signal from the pads.
            Defaults to ``'ecg_pads'``.

        Returns
        -------
        None.
        Adds three labels 'rosc_prediction', 'rosc_probability', and
        'rosc_decision_function'. 'rosc_decision_function' is a
        pseudo-probability computed from the decision function, and
        'rosc_prediction' is the binary prediction.
        """

        if isinstance(cpr_acceleration_source, str):
            cpr_acceleration_source = self.get_channel(cpr_acceleration_source)

        if not isinstance(cpr_acceleration_source, Channel):
            raise ValueError(
                "No valid accelerometer channel specified. Can not predict circulation. "
                "Please specify a suitable channel directly or by name."
            )

        if isinstance(ecg_pads_source, str):
            ecg_pads_source = self.get_channel(ecg_pads_source)

        if not isinstance(ecg_pads_source, Channel):
            raise ValueError(
                "No valid ECG channel specified. Can not predict circulation. "
                "Please specify a suitable channel directly or by name."
            )

        if "cc_periods" not in self.get_label_names():
            self.find_CC_periods_acc(accelerometer_channel=cpr_acceleration_source)

        label_cc_periods = self.data.get_label("cc_periods")
        for label in predict_rosc(
            cpr_acceleration_source, ecg_pads_source, label_cc_periods
        ):
            self.data.add_global_label(label)

    def area_under_threshold(
        self,
        source: Channel | Label | str,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
        threshold: int = 0,
    ) -> ThresholdMetrics:
        """Calculates the area and duration where the signal falls
        below a specified threshold.

        The calculations might be used with a mean arterial pressure to asses for hypotension.
        They are implemented following the proposed metrics by Maheswari et al.
        in :cite:`10.1213/ANE.0000000000003482`.

        See also
        --------
        :func:`.analysis.threshold.area_under_threshold`

        Parameters
        ----------
        source
            The name of a label or channel as a string, which is passed
            to meth:`.get_channel_or_label`, or a :class:`.Channel` or
            :class:`.Label` object.
        start_time
            Start time for truncating the timeseries (meth:`truncate`).
            If ``None``, starts from the beginning.
        stop_time
            End time for truncating the timeseries (meth:`truncate`).
            If ``None``, goes until the end.
        threshold
            The threshold value relative to which the area under the curve (AUC) is calculated.
            Specifically, it computes the area where the signal lies *below* this threshold.

        Returns
        -------
        :class:`.ThresholdMetrics`
        """
        if start_time is None:
            start_time = self.rec_start()
        if stop_time is None:
            stop_time = self.rec_stop()

        if isinstance(source, str):
            source = self.get_channel_or_label(source)

        if not isinstance(source, (Channel, Label)):
            raise ValueError(
                f"The specified source {source} is neither a "
                "Channel or Label, nor the name of a contained "
                "Channel or Label"
            )

        source_data = source.get_data()
        timeseries = pd.Series(source_data.data, index=source_data.time_index)

        return area_under_threshold(
            timeseries=timeseries,
            start_time=start_time,
            stop_time=stop_time,
            threshold=threshold,
        )

    def _prepare_named_outputs(
        self,
        *,
        channel_names: list[str],
        label_names: list[str],
        overwrite_existing_output: bool,
    ) -> None:
        """Validate or remove pre-existing named outputs.

        Parameters
        ----------
        channel_names
            Names of channels produced by the computation.
        label_names
            Names of labels produced by the computation.
        overwrite_existing_output
            If ``True``, all matching outputs are removed before new ones are
            created. Otherwise, a :class:`ValueError` is raised if matching
            outputs already exist.
        """
        existing_channel_names = sorted(
            {
                channel.name
                for name in channel_names
                for channel in self.data.get_channels(name=name)
            }
        )
        existing_label_names = sorted(
            {
                label.name
                for name in label_names
                for label in self.data.get_labels(name=name)
            }
        )

        existing_outputs = existing_channel_names + existing_label_names
        if existing_outputs and not overwrite_existing_output:
            raise ValueError(
                "Outputs already exist in Vitals object. "
                f"Use overwrite_existing_output=True to replace: {existing_outputs}"
            )

        for name in channel_names:
            for channel in list(self.data.get_channels(name=name)):
                self.remove_channel(channel=channel)

        for name in label_names:
            for label in list(self.data.get_labels(name=name)):
                self.remove_label(label=label)

    def compute_respiratory_phases(
        self,
        flow: Channel,
        pressure: Channel,
        inspiratory_threshold: float = 6.55,
        expiratory_threshold: float = 0.85,
        add_labels: bool = True,
        add_intermediate_channels: bool = False,
        return_landmarks: bool = False,
        overwrite_existing_output: bool = False,
    ) -> None:
        """
        Derives the respiratory phases from air flow and airway pressure channels
        and adds them as global labels as described in Orlob et al.
        :cite:`10.1016/j.resuscitation.2026.111050`.

        It expects flow in L/min and pressure in cmH₂O.

        The computation is implemented in
        :func:`.analysis.ventilation.phases.detect_respiratory_phases`.
        This function derives the begin of inspiration based on the product of flow and pressure see :func:`.analysis.ventilation.phases.get_inspiration_starts`.
        The begin of expiration is derived from the product of flow and slope of pressure see :func:`.analysis.ventilation.phases.get_expiration_starts`.
        The begin of inspiration and expiration as well as the inspiration and expiration intervals are added as global labels to the Vitals object.
        Additionally, an interval label for the inspiration phase is created.

        Parameters
        ----------
        self: :class:`.Vitals`
            The Vitals object to which the respiratory phases will be added.

        flow: :class:`.Channel`
            The air flow channel.

        pressure: :class:`.Channel`
            The airway pressure channel.

        inspiratory_threshold: float
            The threshold for detecting the start of inspiration.

        expiratory_threshold: float
            The threshold for detecting the start of expiration.

        add_labels: bool
            Whether to add the derived respiratory phases as labels to the Vitals object.
            If True, the following labels are added:
            - Inspiration Begin as :class:`.Label`
            - Expiration Begin as :class:`.Label`
            - Inspiration as :class:`.IntervalLabel`
            - Expiration as :class:`.IntervalLabel`

        add_intermediate_channels: bool
            Whether to add intermediate channels to the Vitals object. If True, the following channels are added:
            - interpolated flow and pressure
            - product of flow and pressure (power of ventilation) in J/min
            - slope of pressure in cmH₂O/s
            - product of negative flow and slope of pressure in W/s

        return_landmarks: bool
            Whether to return the computed landmarks.

        overwrite_existing_output: bool
            Whether to overwrite existing respiratory phase labels and intermediate channels if they already exist in the Vitals object. If False and such labels exist, the function will raise an exception to avoid unintentional overwriting.

        Returns
        -------
        None or :class:`.RespPhases`
            A RespPhases object containing the timestamps of computed landmarks to derive the respiratory phases if return_landmarks is True, otherwise None.

        Notes
        -----
        This function and its thresholds are developed and tested on data from the following sensors:
            - Flow: Sensirion SFM3000, recording at ~200Hz in slm
            - Pressure: All Sensors DLVR-L60D, recording at ~500Hz in cmH₂O
        """

        self._prepare_named_outputs(
            channel_names=(
                [
                    "Flow Interpolated",
                    "Pressure Interpolated",
                    "Product Flow Pressure",
                    "Slope Pressure",
                    "Product negative Flow Pressures Slope",
                ]
                if add_intermediate_channels
                else []
            ),
            label_names=(
                [
                    "Inspiration Begin",
                    "Expiration Begin",
                    "Inspiration",
                    "Expiration",
                ]
                if add_labels
                else []
            ),
            overwrite_existing_output=overwrite_existing_output,
        )

        result = detect_respiratory_phases(
            flow,
            pressure,
            inspiratory_threshold=inspiratory_threshold,
            expiratory_threshold=expiratory_threshold,
        )

        if add_labels:
            for label in result.labels():
                self.add_global_label(label)

        if add_intermediate_channels:
            for channel in result.intermediate_channels():
                self.add_channel(channel)

        if return_landmarks:
            return result.landmarks
        else:
            return

    def _require_ventilation_volume_inputs(
        self,
    ) -> tuple[Channel, Channel, IntervalLabel, IntervalLabel]:
        """Return validated inputs required for ventilation-volume computation."""
        if "Flow Interpolated" not in self.get_channel_names():
            raise ValueError(
                "Channel 'Flow Interpolated' not found in Vitals object. "
                "Please compute respiratory phases with add_intermediate_channels=True first."
            )

        if "Pressure Interpolated" not in self.get_channel_names():
            raise ValueError(
                "Channel 'Pressure Interpolated' not found in Vitals object. "
                "Please compute respiratory phases with add_intermediate_channels=True first."
            )

        if (
            "Inspiration" not in self.get_label_names()
            or "Expiration" not in self.get_label_names()
        ):
            raise ValueError(
                "Labels 'Inspiration' and 'Expiration' not found in Vitals object. "
                "Please compute respiratory phases with add_labels=True first."
            )

        flow = self.get_channel("Flow Interpolated")
        pressure = self.get_channel("Pressure Interpolated")
        inspiration = self.get_label("Inspiration")
        expiration = self.get_label("Expiration")

        assert isinstance(inspiration, IntervalLabel)
        assert isinstance(expiration, IntervalLabel)
        return flow, pressure, inspiration, expiration

    def compute_ventilation_volumes(
        self,
        correction_factor: float = 1.0,
        overwrite_existing_output: bool = False,
    ) -> None:
        """Compute ventilation-related volume, pressure, and reverse-flow outputs.

        This method expects respiratory phases to have been computed beforehand
        with both intermediate channels and phase labels added to the
        :class:`Vitals` object. It derives breath timing labels, inspiratory
        pressure summaries, net and phase-specific volume channels, cumulative
        inspiratory/expiratory volumes, and reverse-airflow summaries as
        described in :cite:`10.1016/j.resuscitation.2025.110511`.

        Parameters
        ----------
        correction_factor
            Multiplicative factor applied to flow before all integrations. This
            can be used for unit conversion or sensor-specific correction.
        overwrite_existing_output
            If ``True``, previously derived ventilation outputs with the same
            names are removed before recomputation. Otherwise a
            :class:`ValueError` is raised if such outputs already exist.

        Returns
        -------
        None
            The derived channels and labels are attached to the current
            :class:`Vitals` object.
        """
        self._prepare_named_outputs(
            channel_names=[
                "Volume",
                "Inspiratory Volume",
                "Expiratory Volume",
                "Cumulative Inspiratory Volume",
                "Cumulative Expiratory Volume",
            ],
            label_names=[
                "Inspiratory Time",
                "Expiratory Time",
                "Respiratory Rate",
                "Maximal Inspiratory Airway Pressure",
                "Minimal Inspiratory Airway Pressure",
                "Delta VT",
                "VTinsp",
                "VTexp",
                "VTinsp_cum",
                "VTexp_cum",
                "Inspiratory Reverse Airflow Segment Volume",
                "Expiratory Reverse Airflow Segment Volume",
                "Inspiratory Reverse Airflow",
                "Expiratory Reverse Airflow",
                "Inspiratory Reverse Airflow Sum per Inspiration",
                "Expiratory Reverse Airflow Sum per Expiration",
            ],
            overwrite_existing_output=overwrite_existing_output,
        )

        flow, pressure, inspiration, expiration = (
            self._require_ventilation_volume_inputs()
        )

        insp_t = inspiration.get_data().time_index

        ti, te, rr = compute_breath_duration_and_rate_labels(
            inspiration,
            expiration,
        )
        self.add_global_label(ti)
        self.add_global_label(te)
        self.add_global_label(rr)

        p_insp_max, p_insp_min = compute_inspiratory_pressure_labels(
            pressure,
            insp_t,
        )
        pressure.attach_label(p_insp_max)
        pressure.attach_label(p_insp_min)

        volume, v_insp, volume_exp, _, _, _ = compute_volume_channels_and_labels(
            flow,
            inspiration,
            expiration,
            correction_factor,
        )
        self.add_channel(volume)
        self.add_channel(v_insp)
        self.add_channel(volume_exp)

        v_insp_cum, v_exp_cum, _, _ = compute_cumulative_volume_channels(
            flow,
            inspiration,
            correction_factor,
        )
        self.add_channel(v_insp_cum)
        self.add_channel(v_exp_cum)

        (
            insp_reverse_volume,
            insp_reverse_airflow,
            insp_reverse_sum,
            exp_reverse_volume,
            exp_reverse_airflow,
            exp_reverse_sum,
        ) = compute_reverse_airflow_labels(
            flow,
            inspiration,
            expiration,
            correction_factor,
        )

        self.add_global_label(insp_reverse_airflow)
        self.add_global_label(exp_reverse_airflow)
        self.add_global_label(insp_reverse_volume)
        self.add_global_label(exp_reverse_volume)
        self.add_global_label(insp_reverse_sum)
        self.add_global_label(exp_reverse_sum)

        return

    def filter_by_intervallabel(
        self,
        channel_or_label_to_filter: Channel | Label,
        filter_by: IntervalLabel,
        invert: bool = False,
        start_inclusive: bool = True,
        end_inclusive: bool = True,
        full_cover: bool = False,
    ) -> Channel | Label:
        """Filters a channel or label by a given interval label.

        This creates a new channel or label containing only the data points
        that fall within the intervals specified in the given interval label.

        Parameters
        ----------
        channel_or_label_to_filter
            The channel or label to filter.
        filter_by
            The interval label to filter by.
        invert
            If ``True``, inverts the filter, i.e., keeps the parts
            outside the intervals defined by `filter_by`.
        start_inclusive
            If ``True``, the start of the intervals in `filter_by`
            is considered inclusive for filtering.
            If `invert` is ``True``, this parameter becomes inverted too.
        end_inclusive
            If ``True``, the end of the intervals in `filter_by`
            is considered inclusive for filtering.
            If `invert` is ``True``, this parameter becomes inverted too.
        full_cover
            Only relevant for interval Labels:
            - If True, keep label intervals only if they are **fully contained** in at least one filter interval.
            - If False, keep label intervals if they **overlap at all** with any filter interval.

        Returns
        -------
        Channel | Label
            The filtered channel or label.

        See :func:`.analysis.intervals.filter_by_intervallabel`.
        """
        return filter_by_intervallabel(
            channel_or_label_to_filter,
            filter_by,
            invert=invert,
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
            full_cover=full_cover,
        )

    def make_cprdat_analysis(self, model="") -> dict:
        """Automatically analyses a defibrillator recording and and returns a dictionary with all derived information.
        This functions is used in the German Resuscitation Registry to generate a case timeline as interactive visualisation.

        The required analyses (chest compression periods, etCO₂ and ventilations,
        ROSC prediction) add their results as labels to this recording.
        See :func:`.registry.cprdat.make_cprdat_analysis`.
        """
        from vitabel.registry.cprdat import make_cprdat_analysis

        return make_cprdat_analysis(self, model=model)
