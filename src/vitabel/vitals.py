"""Core module, contains the central data container class :class:`Vitals`."""

from __future__ import annotations

import json
import os


import numpy as np
import pandas as pd
import scipy.signal as sgn
import logging
import vitaldb

from typing import Any, Literal

from IPython.display import display
from pathlib import Path

from vitabel.timeseries import (
    Channel,
    Label,
    IntervalLabel,
    TimeDataCollection,
    _timeseries_list_info,
)
from vitabel.utils import (
    loading,
    constants,
    helpers,
    rename_channels,
    predict_circulation,
    construct_snippets,
    deriv,
    av_mean,
    NumpyEncoder,
    determine_gaps_in_recording,
    linear_interpolate_gaps_in_recording,
)
from vitabel.utils import DEFAULT_PLOT_STYLE
from vitabel.typing import (
    Timedelta,
    Timestamp,
    ChannelSpecification,
    LabelSpecification,
    ThresholdMetrics
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

    def add_defibrillator_recording(self, filepath: Path | str, metadata={}) -> None:
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

        The actual loading routines are implemented in the :mod:`.utils.loading` module.

        Parameters
        ----------
        filepath
            The path to a file exported from the defibrillator. Check the
            description above to see, depending on the device type, which
            file path should be passed.
        metadata
            Metadata to be added to the imported data.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError("File not found in directory. Check path!")
        file_extension = filepath.suffix
        filename = filepath.stem
        dirpath = filepath.parent

        logger.info(f"Reading file {filename}")
        fileend_c = [file_extension]
        for file in dirpath.glob("*"):
            if filename in str(file):
                extension = file.suffix
                fileend_c.append(extension)

        logger.info(f"Endings {fileend_c}")

        if ".json" in fileend_c:
            pat_dat, dats = loading.read_zolljson(filepath)
            dats = rename_channels(dats, constants.zoll2channelnames_dict)
            logger.info(f"File: {filename} successfully read!")
        elif (".xml" in fileend_c) and (".txt" not in fileend_c):
            if "_Continuous" in filename:
                pure_filename = filename[: filename.index("_Continuous")]
                fp1 = dirpath.joinpath(Path(pure_filename + "_Continuous.xml"))
                fp2 = dirpath.joinpath(Path(pure_filename + "_Continuous_Waveform.xml"))
                fp3 = dirpath.joinpath(Path(pure_filename + "_CprEventLog.xml"))

                further_files = []
                for file in dirpath.glob("*.xml"):
                    file = str(file)
                    if (
                        pure_filename in file
                        and "_Continuous.xml" not in file
                        and "_Continuous_Waveform.xml" not in file
                        and "_CprEventLog.xml" not in file
                        and ".xml" in file
                    ):
                        file_ending = file[
                            file.find(filename) + len(filename) : file.rfind(".")
                        ]
                        further_files.append(file_ending)

                if further_files:
                    logger.warning(
                        f"Warning! Further Lifepak-Files {further_files} for recording {pure_filename} found in directory. These files are not loaded to the cardio-Object currently. Please modify the loading routine to include them"
                    )
                if not (
                    os.path.isfile(fp1) and os.path.isfile(fp2) and os.path.isfile(fp3)
                ):
                    raise FileNotFoundError(
                        f"Error when Loading LIFEPAK Recording! Expected Files {fp2} and {fp3}, which are not found, and the recording cannot be loaded. Check Export of your LIFEPAK_Recording"
                    )

                else:
                    pat_dat, dats = loading.read_lifepak(
                        fp1, fp2, fp3, further_files=further_files
                    )
                    dats = rename_channels(dats, constants.LP2channelnames_dict)
                logger.info(f"LIFEPAK-File: {str(filename)} successfully read!")
            elif "_Lucas" in filename:
                pure_filename = filename[: filename.index("_Lucas")]

                fp1 = dirpath.joinpath(Path(pure_filename + "_Lucas.xml"))
                fp2 = dirpath.joinpath(Path(pure_filename + "_CprEventLog.xml"))
                if not (os.path.isfile(fp1) and os.path.isfile(fp2)):
                    raise FileNotFoundError(
                        f"Error when Loading LUCAS Recording! Expected Files additional file {fp2}, which is not found, and the recording cannot be loaded. Check Export of your LUCAS-Recording"
                    )

                else:
                    pat_dat, dats = loading.read_lucas(fp1, fp2)
                    dats = rename_channels(dats, constants.LP2channelnames_dict)

                if dats:
                    logger.info(f"LUCAS-File: {str(filename)} successfully read!")
                else:
                    logger.warning(f"LUCAS-File: {str(filename)} is empty!")
                    return None

            else:
                pat_dat, dats = loading.read_zollxml(
                    dirpath.joinpath(Path(filename + ".xml"))
                )
                dats = rename_channels(dats, constants.zoll2channelnames_dict)

                logger.info(f"File: {filename}.xmlsuccessfully read!")
        elif ".txt" in fileend_c:
            pure_filename = filename[: filename.index("_ecg")]

            pat_dat, dats = loading.read_zollcsv(
                dirpath.joinpath(Path(pure_filename + "_ecg.txt")),
                dirpath.joinpath(Path(pure_filename + ".xml")),
            )
            dats = rename_channels(dats, constants.zoll2channelnames_dict)

            logger.info(f"File: {filename} successfully read!")
        elif ".bdf" in fileend_c:
            pat_dat, dats = loading.read_corpuls(
                dirpath.joinpath(Path(filename + ".bdf"))
            )
            dats = rename_channels(dats, constants.corpuls2channelnames_dict)

            logger.info(f"File: {filename} successfully read!")

        elif fileend_c != []:
            logger.error(f"Error: No method to read {fileend_c} files!")
            return None

        if pat_dat:
            for key in ["File Name", "File ID", "Serial Nr", "Model"]:
                if key in pat_dat["Main data"].index:
                    metadata[key] = pat_dat["Main data"].loc[key].iloc[0]
            if "Model" in pat_dat["Main data"].index:
                metadata["source"] = pat_dat["Main data"].loc["Model"].iloc[0]

        # Convert the data into channels and add to the channel class.
        self.dats = dats
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

    def add_vital_db_recording(
        self,
        filepath: Path | str,
        metadata={"source": "VitalDB-Recording"},
    ) -> None:
        """Loading channels from a vitalDB recording.

        Parameters
        ----------
        filepath
            The path to the recording. Must be a ``*.vit`` file.
        """
        vit = vitaldb.VitalFile(str(filepath))
        df = vit.to_pandas(vit.get_track_names(), interval=None, return_datetime=True)
        df.set_index("Time", inplace=True, drop=True)
        self.add_data_from_DataFrame(df, metadata=metadata)

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
                            label_dict[label]["timestamp"], ann_dict["One-Annotator"][label]
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
        if not ("timestamp" in source.keys() and "data" in source.keys()):
            raise ValueError(
                "The dictionary must contain a 'timestamp' and a 'data' key which contain timestamps and data for this channel. \n \
                             In case of time_only or 'time_interval' channel_types choose 'data' to be an empty list."
            )
    
        if time_start:
            time_start = pd.Timestamp(time_start)

        time = source["timestamp"]
        data = source["data"]
        if len(time) == 0:
            logger.warning("The data is empty. No channel or label will be added.")
            return

        if datatype == "channel":
            channel = Channel(
                name,
                time,
                data,
                metadata=metadata,
                time_start=time_start,
            )
            self.data.add_channel(channel)
        elif datatype == "label":
            label = Label(
                name,
                time,
                data,
                metadata=metadata,
                time_start=time_start,
                anchored_channel=anchored_channel,
            )
            if anchored_channel is None:
                self.data.add_global_label(label)
        elif datatype == "interval_label":
            label = IntervalLabel(
                name,
                time,
                data,
                metadata=metadata,
                time_start=time_start,
                anchored_channel=anchored_channel,
            )
            if anchored_channel is None:
                self.data.add_global_label(label)

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
        if isinstance(source, Path):
            with open(source, "r") as file:
                source = json.load(file)

        for key in source:
            if not isinstance(source[key], dict):
                raise ValueError(
                    "Source must be a dictionary of the form "
                    "{'channel1': {'timestamp': [...], 'data': [...]},"
                    " 'channel2': { ... }, ... }. "
                    f"For key {key} the value is not a dict."
                )
            else:
                self._add_single_dict(
                    source[key],
                    key,
                    metadata=metadata,
                    time_start=time_start,
                    datatype=datatype,
                    anchored_channel=anchored_channel,
                )

    def add_data_from_DataFrame(
        self,
        source: pd.DataFrame,
        time_start: str | None = None,
        time_unit=None,
        datatype: Literal["channel", "label", "interval_label"] = "channel",
        anchored_channel: Channel | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Adds Data from a ``pandas.DataFrame``.

        Parameters
        ----------
        source
            The DataFrame containing the data. The index of the DataFrame contains the
            time (either as DatetimeIndex or numeric Index),
            and the columns contain the channels. NaN-Values in the columns are
            not taken into account an ignored.
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
            A dictionary containing all the metadata for the channels / labels.

        Raises
        ------
        ValueError
            The DataFrame does not contain a DateTime or Numeric Index.
        """

        if not (
            isinstance(source.index, pd.DatetimeIndex)
            or (pd.api.types.is_numeric_dtype(source.index))
        ):
            raise ValueError(
                "The DataFrame needs to have a datetime or a numeric index, "
                "which describes the time of the timeseries."
            )
        
        for col in source.columns:
            series = source[col]
            series = series[series.notna()]
            time = np.array(series.index)
            data = series.values
            if len(time) == 0:
                continue

            if datatype == "channel":
                channel = Channel(
                    name=col,
                    time_index=time,
                    data=data,
                    time_start=time_start,
                    time_unit=time_unit,
                    metadata=metadata,
                )
                self.data.add_channel(channel)
            elif datatype == "label":
                label = Label(
                    col,
                    time,
                    data,
                    time_start=time_start,
                    time_unit=time_unit,
                    metadata=metadata,
                    anchored_channel=anchored_channel,
                )
                if anchored_channel is None:
                    self.data.add_global_label(label)
            elif datatype == "interval_label":
                raise NotImplementedError("Interval labels can currently not be added from DataFrames.")

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
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        df = pd.read_csv(file_path, **kwargs)
        self.add_data_from_DataFrame(
            df,
            time_start=time_start,
            time_unit=time_unit,
            datatype=datatype,
            anchored_channel=anchored_channel,
            metadata=metadata,
        )

    def add_channel(self, Channel):
        self.data.add_channel(Channel)

    def add_global_label(self, Label):
        self.data.add_global_label(Label)

    # # Add loading iterable of Frame or Series to add more channels at once
    # time_unit = time_unit,
    def remove_channel(self, *, channel: Channel | None = None, **kwargs):
        self.data.remove_channel(channel=channel, **kwargs)

    def remove_label(self, *, label: Label | None = None, **kwargs):
        self.data.remove_label(label=label, **kwargs)

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
        json_dict = {"metadata": self.metadata, "data": data_dict, "hash": data_hash}

        with open(path, "w") as fd:
            json.dump(json_dict, fd, cls=NumpyEncoder)

    def load_data(self, path: Path | str, check_channel_hash=True):
        """
        Loads all Channels and Labels from a previously saved JSON file.

        Parameters
        ----------
        path
            The Path of the channel data.
        """
        from vitabel import __version__

        path = Path(path)
        with open(path, "r") as fd:
            json_dict = json.load(fd)
        self.metadata = json_dict["metadata"]
        self.data = TimeDataCollection.from_dict(json_dict["data"])
        saved_hash = json_dict["hash"]
        channel_hash = self.data.channel_data_hash()
        vitabel_version = json_dict["metadata"].get("vitabel version", "unknown")
        if vitabel_version != "unknown" and __version__ != vitabel_version:
            logger.warning(
                f"The vitabel version used to save the data is {vitabel_version}, "
                f"while the currently installed version is {__version__}. "
                "This may lead to compatibility issues."
            )

        if check_channel_hash:
            saved_hash = channel_hash
            if channel_hash != saved_hash:
                raise ValueError(
                    "The hash value has changed, which indicates that the data "
                    "has been modified outside of the Vitals class. Aborting loading. "
                    "Pass check_channel_hash=False to ignore this check."
                )

            # Add Offsets from metadata labels the channel json contains the raw data, without offset correction

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Retrieve data -------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def get_channels(self, name: str | None = None, **kwargs) -> list[Channel]:
        return self.data.get_channels(name, **kwargs)

    def get_channel(self, name: str | None = None, **kwargs) -> Channel:
        return self.data.get_channel(name, **kwargs)

    def get_labels(self, name: str | None = None, label_type: type[Label] | None = None, **kwargs) -> list[Label]:
        """Returns a list of labels based on their name.
        
        Parameters
        ----------
        name : str
            The name of the labels to retrieve.
            See :meth:`.get_label` for valid specifications.
        label_type : TYPE, optional
            A specification of the label type (IntervalLabel or Label) to retrieve
        kwargs
            Keyword arguments to filter the labels by. 
            See :meth:`.get_label` for valid specifications
        """

        if label_type is not None:
            return [label for label in self.get_labels(name) if type(label) is label_type]
        return self.data.get_labels(name, **kwargs)

    def get_label(self, name: str | None = None, **kwargs) -> Label:
        """Return a list of labels.

        See also
        --------
        :meth:`.TimeDataCollection.get_label`

        Parameters
        ----------
        name
            The name of the label to retrieve. Allowed to be passed
            either as a positional or a keyword argument.
        kwargs
            Keyword arguments to filter the labels by. The
            specified arguments are compared to the attributes
            of the labels.
        """
        return self.data.get_label(name, **kwargs)

    def get_channels_or_labels(
        self, name: str | None = None, label_type: type[Label] | None = None,  **kwargs
    ) -> list[Channel | Label]:
        return self.data.get_channels(name, **kwargs) + self.get_labels(
            name, label_type=label_type, **kwargs
        )

    def get_channel_or_label(
        self, name: str | None = None, label_type: type[Label] | None = None, **kwargs
    ) -> Channel | Label:
        channels_or_labels = self.get_channels_or_labels(name, label_type=label_type, **kwargs)
        if len(channels_or_labels) != 1:
            raise ValueError(
                "Channel or Label specification was ambiguous, no unique channel or Label "
                f"was identified. Query returned: {channels_or_labels}"
            )
        return channels_or_labels[0]

    def get_channel_names(self) -> list[str]:  # part of register application
        """Return a list with the names of all channels in the recording."""
        return self.data.channel_names

    def get_label_names(self) -> list[str]:  # part of register application
        """Returns a list with the names of all labels."""
        return self.data.label_names

    def get_channel_or_label_names(self) -> list[str]:
        """Returns a list with all names from either channels or labels."""
        return self.get_channel_names() + self.get_label_names()

    def keys(self) -> list[str]:
        """Alias for :meth:`get_channel_or_label_names`."""
        return self.get_channel_or_label_names()

    def rec_start(self) -> pd.Timestamp | None:  # part of register application
        """Returns the first timestamp among all channels in this case or None if no channel exists."""
        if not self.channels:
            return None
        if self.data.is_time_absolute():
            start_time = self.data.channels[0].time_start
            for chan in self.channels:
                if chan.time_start < start_time:
                    start_time = chan.time_start
        return start_time

    def rec_stop(self) -> pd.Timestamp | None:  # part of register application
        """Returns the last timestamp among all channels in this case or None if no channel exists."""
        if not self.channels:
            return None
        if self.data.is_time_absolute():
            stop_time = (
                self.data.channels[0].time_start + self.data.channels[0].time_index[-1]
            )
            for chan in self.channels:
                cha_stop_time = chan.time_start + chan.time_index[-1]
                if cha_stop_time > stop_time:
                    stop_time = cha_stop_time
        return stop_time

    def get_channel_infos(self) -> pd.DataFrame:
        """Returns information about all channels in a nicely formatted
        ``pandas.DataFrame``.
        """
        return _timeseries_list_info(self.channels)

    def get_label_infos(self) -> pd.DataFrame:
        """Returns information about all labels in a nicely formatted
        ``pandas.DataFrame``.
        """
        return _timeseries_list_info(self.labels)

    def info(self) -> None:
        """Displays relevant information about the channels and labels
        currently present in the recording.
        """
        channel_info = self.get_channel_infos()
        label_info = self.get_label_infos()
        display(channel_info)
        display(label_info)

    def truncate(
        self,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
    ) -> Vitals:
        """Return a new object where time data has been truncated to a specified interval.

        Parameters
        ----------
        start_time
            The start time of the truncated recording.
        stop_time
            The stop time of the truncated recording.

        Returns
        -------
        Vitals
            A new Vitals object containing the truncated recording.
        """
        truncated_vitals = Vitals()
        for channel in self.channels:
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
        resolution
            The resolution of the plot in the time unit of the channels.
            If not specified, the channel and label data is not downsampled.
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
        """Return a list of all defibrillations with all stored information
        in a DataFrame.

        Returns
        -------
        shocks : pandas.DataFrame
            DESCRIPTION.

        """
        defib_data = {}
        defib_channel_names = [
            channel_name
            for channel_name in self.get_channel_names()
            if "defibrillations" in channel_name
        ]

        for channel in self.channels:
            if channel.name in defib_channel_names:
                time, data = channel.get_data()
                defib_data[channel.name] = {"timestamp": time, "data": data}

        time_index = np.array([])
        all_times_equal = True
        for key in defib_data:
            if not time_index.any():
                time_index = np.array(defib_data[key]["timestamp"])
            all_times_equal = (
                all_times_equal & (defib_data[key]["timestamp"] == time_index).all()
            )

        if all_times_equal:
            data = np.asarray([defib_data[key]["data"] for key in defib_data]).T
            keys = [key[key.find("_") + 1 :] for key in defib_data]
            if data.any():
                shocks = pd.DataFrame(data, index=time_index, columns=keys)
                shocks.index.name = "timestamp"
            else:
                shocks = pd.DataFrame()
            return shocks

        else:
            logger.error(
                "Error. Different Defibrillation keys contain different timestamp. Cannot construct single DataFrame from this"
            )
            return None

    def compute_etco2_and_ventilations(
        self,
        mode: Literal['threshold', 'filter'] = 'filter',
        breath_threshold: float = 2,
        etco2_threshold: float = 3,
        **kwargs,
    ):
        """
        Computes end-tidal CO2 (etCO₂) values and timestamps of ventilations from the capnography waveform,
        and adds them as labels.

        The capnography signal must be present as a channel named 'capnography'. Two detection methods are supported:

        - 'filter': An unpublished method by Wolfgang Kern (default).
        - 'threshold': The method described by Aramendi et al., "Feasibility of the capnogram to monitor ventilation rate during cardiopulmonary resuscitation"
          (Resuscitation, 2016, DOI: `10.1016/j.resuscitation.2016.08.033 <https://doi.org/10.1016/j.resuscitation.2016.08.033>`_).

        Parameters
        ----------
        mode
            Method to use for detecting ventilations from the CO₂ signal.

            - 'filter': An unpublished method by Kern (default)
            - 'threshold': The method described by Aramendi et al.

        breath_threshold
            Threshold below which a minimum is identified as a ventilation (default: 2 mmHg). Used by the 'filter' method.
        etco2_threshold
            Threshold above which a maximum is identified as an etCO₂ value of an expiration (default: 3 mmHg). Used by the 'filter' method.
        """
        # Support legacy parameter name
        if 'breaththresh' in kwargs:
            if breath_threshold is not None:
                raise TypeError("Cannot specify both 'breath_threshold' and legacy 'breaththresh'")
            breath_threshold = kwargs.pop('breaththresh')
            logger.warning(
                "The keyword argument breaththresh is deprecated, "
                "use breath_threshold instead"
            )

        if "capnography" not in self.get_channel_names():
            logger.error(
                "Error! No Capnography Signal found. Cannot compute etCO₂ and detect ventilations"
            )
        else:
            co2_channel = self.data.get_channel("capnography")
            cotime, co = co2_channel.get_data()  # get data

            freq = np.timedelta64(1, "s") / np.nanmedian(cotime.diff())
            cotime = np.asarray(cotime)
            co = np.asarray(co)
            if mode == "filter":  # Wolfgang Kern's unpublished method
                but = sgn.butter(4, 1 * 2 / freq, btype="lowpass", output="sos")
                co2 = sgn.sosfiltfilt(but, co)  # Filter forwarsd and backward
                et_index = sgn.find_peaks(co2, distance=1 * freq, height=etco2_threshold)[
                    0
                ]  # find peaks of filtered signal as markers for etco2
                resp_index = sgn.find_peaks(
                    -co2, distance=1 * freq, height=-breath_threshold
                )[  # find dips of filtered signal as markers for ventilations
                    0
                ]

                etco2time = cotime[et_index]  # take elements on this markers
                etco2 = co[et_index]
                resptime = cotime[resp_index]
                resp_height = co[resp_index]

                # initialize search for other markers
                k = 0
                del_resp = []
                more_resp_flag = False
                min_resp_height = np.nan
                min_resp_index = np.nan

                # look a signal before first ventilation

                co2_maxtime = etco2time[(etco2time < resptime[0])]
                co2_max = etco2[(etco2time < resptime[0])]
                netco2 = len(co2_maxtime)

                # when there is more than a single maximum before first respiration, take only largest one
                if netco2 > 1:
                    k_max = np.argmax(co2_max)
                    for j in range(k + k_max, k, -1):
                        etco2 = np.delete(etco2, k)
                        etco2time = np.delete(etco2time, k)
                    for j in range(k + netco2, k + k_max + 1, -1):
                        etco2 = np.delete(etco2, k + 1)
                        etco2time = np.delete(etco2time, k + 1)
                    k += 1

                # if there is no maximum
                elif netco2 == 0:
                    pass
                # if there is a single maximum
                else:
                    k += 1

                for i, resp in enumerate(resptime[:-1]):
                    next_resp = resptime[i + 1]
                    # check maxima until next respiration same as bevfore
                    co2_maxtime = etco2time[
                        (etco2time >= resp) & (etco2time < next_resp)
                    ]
                    co2_max = etco2[(etco2time >= resp) & (etco2time < next_resp)]
                    netco2 = len(co2_maxtime)
                    if netco2 > 1:  # take largest one
                        k_max = np.argmax(co2_max)
                        for j in range(k + k_max, k, -1):
                            etco2 = np.delete(etco2, k)
                            etco2time = np.delete(etco2time, k)
                        for j in range(k + netco2, k + k_max + 1, -1):
                            etco2 = np.delete(etco2, k + 1)
                            etco2time = np.delete(etco2time, k + 1)
                        k += 1
                        more_resp_flag = False
                    elif netco2 == 0:
                        if more_resp_flag:
                            if resp_height[i] > min_resp_height:
                                del_resp.append(i)
                            else:
                                del_resp.append(min_resp_index)
                                min_resp_height = resp_height[i]
                        else:
                            if resp_height[i] > resp_height[i + 1]:
                                del_resp.append(i)
                                min_resp_height = resp_height[i + 1]
                                min_resp_index = i + 1

                                more_resp_flag = True
                            else:
                                del_resp.append(i + 1)
                                min_resp_height = resp_height[i]
                                min_resp_index = i
                                more_resp_flag = True

                    else:
                        more_resp_flag = False
                        k += 1

                del_resp.sort()
                for i in del_resp[::-1]:
                    resptime = np.delete(resptime, i)

            elif mode == "threshold":   # Aramendi et al., 2016
                but = sgn.butter(4, 10 * 2 / freq, btype="lowpass", output="sos")
                co2 = sgn.sosfiltfilt(but, co)  # Filter forwarsd and backward
                d = freq * (co2[1:] - co2[:-1])
                exp_index2 = sgn.find_peaks(d, height=0.35 * freq)[0]
                ins_index2 = sgn.find_peaks(-d, height=0.45 * freq)[0]

                final_flag = False
                ins_index3 = []
                exp_index3 = []
                j_ins = 0
                j_exp = 0
                while not final_flag:
                    ins_index3.append(ins_index2[j_ins])
                    while exp_index2[j_exp] < ins_index2[j_ins]:
                        j_exp += 1
                        if j_exp == len(exp_index2) - 1:
                            final_flag = True
                            break
                    exp_index3.append(exp_index2[j_exp])
                    while ins_index2[j_ins] < exp_index2[j_exp]:
                        j_ins += 1
                        if j_ins == len(ins_index2) - 1:
                            final_flag = True
                            break

                resptime = []
                etco2time = []
                etco2 = []
                Th1_list = [5 for i in range(5)]
                Th2_list = [0.5 for i in range(5)]
                Th3_list = [0 for i in range(5)]

                k = 0
                for i_ins, i_exp, i_next_ins in zip(
                    ins_index3[:-1], exp_index3[:-1], ins_index3[1:]
                ):
                    D = (i_exp - i_ins) / freq
                    A_exp = 1 / (i_next_ins - i_exp) * np.sum(co2[i_exp:i_next_ins])
                    A_ins = 1 / (freq * D) * np.sum(co2[i_ins:i_exp])
                    A_r = (A_exp - A_ins) / A_exp
                    S = 1 / freq * np.sum(co2[i_exp : i_exp + int(freq)])
                    if len(resptime) > 0:
                        t_ref = pd.Timedelta(
                            (cotime[i_exp] - resptime[-1])
                        ).total_seconds()
                    else:
                        t_ref = 2  # if t_ref >1.5 then it is ok, so 2 does the job
                    if D > 0.3:
                        if (
                            A_exp > 0.4 * np.mean(Th1_list)
                            and A_r > np.min([0.7 * np.mean(Th2_list), 0.5])
                            and S > 0.4 * np.mean(Th3_list)
                        ):
                            if t_ref > 1.5:
                                resptime.append(cotime[i_exp])
                                Th1_list[k] = A_exp
                                Th2_list[k] = A_r
                                Th3_list[k] = S
                                etco2time.append(
                                    cotime[i_exp + np.argmax(co2[i_exp:i_next_ins])]
                                )
                                etco2.append(np.max(co2[i_exp:i_next_ins]))
                                k += 1
                                k = k % 5
            if mode == "threshold" or mode == "filter":
                metadata = {
                    "creator": "automatic",
                    "creation_date": pd.Timestamp.now(),
                    "creation_mode": mode,
                }
                etco2_lab = Label(
                    "etco2_from_capnography",
                    time_index=etco2time,
                    data=etco2,
                    metadata=metadata,
                    plotstyle=DEFAULT_PLOT_STYLE.get("etco2_from_capnography", None),
                )
                co2_channel.attach_label(etco2_lab)

                vent_lab = Label(
                    "ventilations_from_capnography",
                    time_index=resptime,
                    data=None,
                    metadata=metadata,
                    plotstyle=DEFAULT_PLOT_STYLE.get(
                        "ventilations_from_capnography", None
                    ),
                )
                co2_channel.attach_label(vent_lab)
            else:
                logger.error(
                    f"mode {mode} not known. Please use either 'filter' or 'threshold' as argument"
                )

    def cycle_duration_analysis(
            self,
            cc_events_channel: Channel | str | None
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
            The method is described in `10.1016/j.resuscitation.2021.12.028 <https://doi.org/10.1016/j.resuscitation.2021.12.028>`_ or in the
            Thesis 'Towards a data-driven cardiac arrest treatment' by Wolfgang Kern in more detail.
            See https://unipub.uni-graz.at/obvugrhs/content/titleinfo/10138095 for more information.
        """
        if isinstance(cc_events_channel, str):
            if cc_events_channel not in self.get_channel_names():
                logger.error(
                    f"The specified channel '{cc_events_channel}' could not be identified."
                    "Please specify a channel or a string with the name of the channel."
                )
                return
            else:
                cc_events_channel = self.get_channel(cc_events_channel)
        elif not isinstance(cc_events_channel, Channel):
            logger.error(
                "No valid channel with chest compression specified. Can not identify CC-periods via single CCs."
                "Please specify a channel or a string with the name of the channel."
            )
            return
        elif cc_events_channel is None:
            available_channels = set(self.get_channel_names()) & {"cc", "cc_depth"}
            if available_channels:
                cc_events_channel = self.get_channel(next(iter(available_channels)))
            else:            
                logger.error(
                    "Could not identify channels with single chest compressions."
                    "Please specify a channel or a string with the name of the channel."
                )
                return
            
        comp, *_ = cc_events_channel.get_data() # get data
        comp = np.sort(comp)

        t_ref = cc_events_channel.time_start

        if cc_events_channel.is_time_relative():
            comp = comp.astype("timedelta64[s]").astype(float)
        else:
            comp = np.array([(t - t_ref).total_seconds() for t in comp]) #TODO: check if times have ti be coerced

        compression_counter = 1  # number of compressions in cc period
        last_c = comp[0]  # initilaize last compression
        sta = np.array([comp[0]])  # start ... = first compression
        sto = np.array([comp[0]])  # stop preliminary = first compression
        fre = np.array([0.6])  # take a cc length of 0.6 s = 100 bpm as start value
        for c in comp[1:]:  # Iterate through all compressions
            if c - last_c < 3 * np.mean(fre[-5:]):
                # If difference from next marker to last marker is smaller than
                # three times average cc length, then compressions are connected
                # (one period)
                if len(fre) == 1:  # Remove initial cc length guess
                    fre = np.array([])
                fre = np.append(
                    fre, c - last_c
                )  # estimate cc length with the actual value
                sto[-1] = c  # stop is new marker

                last_c = c  # reload last compression
                compression_counter += 1
            else:  # If difference betweeen markers is larger 3 times average cc length
                if (
                    compression_counter < 3
                ):  # If less then three compressions delete start and stop markers
                    sta = np.delete(sta, -1)
                    sto = np.delete(sto, -1)
                else:  # If compression period is valid
                    sta[-1] = sta[-1] - np.mean(fre[:5])  # correct starting point with
                sta = np.append(sta, c)
                sto = np.append(sto, c)
                fre = np.array([0.6])
                last_c = c
                compression_counter = 1
        if compression_counter < 3:
            sta = np.delete(sta, -1)
            sto = np.delete(sto, -1)
        else:
            sta[-1] = sta[-1] - np.mean(fre[:5])
        metadata = {
            "creator": "automatic",
            "creation_date": pd.Timestamp.now(),
            "method": "cycle_duration_analysis",
        }

        periods = np.empty(sta.size + sto.size, dtype=sta.dtype)
        periods[0::2] = sta
        periods[1::2] = sto

        cc_periods = IntervalLabel(
            name = "cc_periods",
            time_index = periods, 
            time_start = t_ref, 
            metadata = metadata,
            plot_type = "box",
            plotstyle = DEFAULT_PLOT_STYLE.get("cc_periods", None)
        )

        cc_events_channel.attach_label(cc_periods)

    def find_CC_periods_acc(
            self, 
            accelerometer_channel: Channel | str = 'cpr_acceleration',
        ) -> None:  # part of register application
        """
        Automatically detects periods of continuous chest compressions.

        The procedure is implemented as described in `10.1016/j.resuscitation.2021.12.028 <https://doi.org/10.1016/j.resuscitation.2021.12.028>` and `10.1016/j.dib.2022.107973 <https://doi.org/10.1016/j.dib.2022.107973>`_.
        In essence it uses the root mean square of the accelerometer signal of feedback sensor for cardiopulmonary resuscitation to detect the rise in "power" of the signal linked to the alteration by the accelerations of continous chest compressions. 

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
            The method is described in `10.1016/j.resuscitation.2021.12.028 <https://doi.org/10.1016/j.resuscitation.2021.12.028>`_ or in the
            Thesis 'Towards a data-driven cardiac arrest treatment' by Wolfgang Kern in more detail.
            See https://unipub.uni-graz.at/obvugrhs/content/titleinfo/10138095 for more information.
        """
        if isinstance(accelerometer_channel, str):
            if accelerometer_channel not in self.get_channel_names():
                logger.error(
                    f"The specified channel '{accelerometer_channel}' could not be identified."
                    "Please specify a channel or a string with the name of the channel."
                )
                return
            else:
                ACC_channel = self.get_channel(accelerometer_channel)
        elif not isinstance(accelerometer_channel, Channel):
            logger.error(
                "No valid accelerometer channel specified. Can not identify CC-periods via acceleration."
                "Please specify a channel or a string with the name of the channel."
            )
            return
        else:
            ACC_channel = accelerometer_channel
        #NOTE: Do we need a check if the acclerometer channel specified as channel is a channel of self?

        acctime, acc = ACC_channel.get_data()  # get data
        freq = np.timedelta64(1, "s") / np.nanmedian(acctime.diff())
        t_ref = ACC_channel.time_start

        if ACC_channel.is_time_relative():
            acctime = acctime.astype("timedelta64[s]").astype(float)
        else:
            acctime = np.asarray([(t - t_ref).total_seconds() for t in acctime])

        acctime = np.asarray(acctime)
        acc = np.asarray(acc - np.mean(acc))
        gap_start, gap_stop, gap_start_indices = determine_gaps_in_recording(
            acctime, acc
        )
        acctime, acc = linear_interpolate_gaps_in_recording(acctime, acc)
        but = sgn.butter(
            4, (0.2 * 2 / freq, 50 * 2 / freq), btype="bandpass", output="sos"
        )
        acce = sgn.sosfilt(but, acc)
        window_size = int(freq)  # Good to ignore short pauses
        softthres = 10
        avacc = av_mean(window_size, np.abs(acce))  # Average mean of abs(acc)
        davacc = deriv(acctime, avacc)  # Derivative of average mean
        avdavacc = av_mean(window_size, davacc)  # av_mean of derivative
        # Soft Thresholding to get rid of small extrema in derivative due to oscillations during cpr
        avdavacc = np.maximum(np.abs(avdavacc) - softthres, 0) * np.sign(avdavacc)

        n = len(acctime)
        thresh = 0
        peakmark = (avdavacc[2:] - avdavacc[1:-1]) * (
            avdavacc[1:-1] - avdavacc[:-2]
        )  # Determine peaks in averagre derivative
        pointcand = np.arange(0, n - 2)[
            (peakmark <= 0) & (np.abs(avdavacc[1:-1]) > thresh)
        ]  # possible starting poins
        points = {"Start": np.array([], int), "Stop": np.array([], int)}
        flag = False  # Start with search for starting point

        cand = 0
        icand = 0
        for i in pointcand:
            if (
                not flag and avdavacc[i] < 0
            ):  # while searching for start a stoplike value appears. save start value
                points["Start"] = np.append(points["Start"], icand)
                flag = not flag
                cand = 0
            elif (
                flag and avdavacc[i] > 0
            ):  # while searching for stop a startike value appears. save stop value
                points["Stop"] = np.append(points["Stop"], icand)
                flag = not flag
                cand = 0
            if not flag:  # Searching for start: Get maximum of derivative
                if avdavacc[i] > cand:
                    icand = i
                    cand = avdavacc[icand]
            else:  # Searching for end: Get minimum of derivative
                if avdavacc[i] < cand:
                    icand = i
                    cand = avdavacc[icand]
        if not flag:  # add last point to start or endpoint (not in loop included)
            points["Start"] = np.append(points["Start"], icand)
        else:
            points["Stop"] = np.append(points["Stop"], icand)

        badpoints = np.array([], int)
        for i in range(
            np.minimum(len(points["Start"]), len(points["Stop"])) - 1
        ):  # Delete pauses, where average mean stays over 0.5 * mean(CPR Phase before and CPR phase after)
            pausethresh = (
                0.35
                * 0.5
                * (
                    np.mean(avacc[points["Start"][i] : points["Stop"][i]])
                    + np.mean(avacc[points["Start"][i + 1] : points["Stop"][i + 1]])
                )
            )
            if np.min(avacc[points["Stop"][i] : points["Start"][i + 1]]) > pausethresh:
                badpoints = np.append(badpoints, i)

        points["Start"] = np.delete(points["Start"], badpoints + 1)
        points["Stop"] = np.delete(points["Stop"], badpoints)

        pauselen = (
            1.6  # a CPR phase must last at least 1.6 seconds, delete shorter ones
        )
        badpoints2 = np.array([], int)
        for i in range(
            np.minimum(len(points["Start"]), len(points["Stop"]))
        ):  # Delete CPR-Periods which are shorter then 2.5s
            if acctime[points["Stop"][i]] - acctime[points["Start"][i]] < pauselen:
                badpoints2 = np.append(badpoints2, i)

        points["Start"] = np.delete(points["Start"], badpoints2)
        points["Stop"] = np.delete(points["Stop"], badpoints2)

        cpr_thresh = 28  # 30 #Acc_mean while cpr is not allowed to be below this threshold (unit presumably equivalent 3,5 inch /s^2 = 0.09 m/s^2, but units remain unclear)

        badpoints3 = np.array([], int)
        for i in range(
            np.minimum(len(points["Start"]), len(points["Stop"]))
        ):  # Delete CPR-Periods which are shorter then 2.5s
            if np.mean(avacc[points["Start"][i] : points["Stop"][i]]) < cpr_thresh:
                badpoints3 = np.append(badpoints3, i)

        points["Start"] = np.delete(points["Start"], badpoints3)
        points["Stop"] = np.delete(points["Stop"], badpoints3)
        nlen = int(freq // 2)
        for i in range(len(points["Start"])):
            elem = points["Start"][i]
            if elem > nlen and elem < n - nlen:
                points["Start"][i] = int(
                    np.sum(
                        avdavacc[elem - nlen : elem + nlen]
                        * np.arange(elem - nlen, elem + nlen, 1)
                    )
                    / np.sum(avdavacc[elem - nlen : elem + nlen])
                )
        for i in range(len(points["Stop"])):
            elem = points["Stop"][i]
            if elem > nlen and elem < n - nlen:
                points["Stop"][i] = int(
                    np.sum(
                        avdavacc[elem - nlen : elem + nlen]
                        * np.arange(elem - nlen, elem + nlen, 1)
                    )
                    / np.sum(avdavacc[elem - nlen : elem + nlen])
                )

        if points["Start"][0] > points["Stop"][0]:
            points["Stop"] = np.delete(points["Stop"], 0)
        if points["Start"][-1] > points["Stop"][-1]:
            points["Start"] = np.delete(points["Start"], -1)

        starts = acctime[points["Start"]]
        stops = acctime[points["Stop"]]

        gap_starts_to_append = []
        gap_stops_to_append = []

        for gap_i, gap_f in zip(gap_start, gap_stop):
            if len(starts[starts < gap_i]) > 0:
                if len(stops[stops < gap_i]) == 0:
                    gap_starts_to_append.append(gap_i)
                elif starts[starts < gap_i][-1] > stops[stops < gap_i][-1]:
                    gap_starts_to_append.append(gap_i)
            if len(stops[stops > gap_i]) > 0:
                if len(starts[starts > gap_i]) == 0:
                    gap_stops_to_append.append(gap_f)
                elif starts[starts > gap_i][0] > stops[stops > gap_i][0]:
                    gap_stops_to_append.append(gap_f)

        starts = np.append(starts, gap_stops_to_append)
        stops = np.append(stops, gap_starts_to_append)

        starts = np.sort(starts)
        stops = np.sort(stops)
        #NOTE: unsure if we have to maintain starts and stops, we might even just go wit append and sort
        periods = np.empty(starts.size + stops.size, dtype=starts.dtype)
        periods[0::2] = starts
        periods[1::2] = stops

        metadata = {
            "creator": "automatic",
            "creation_date": pd.Timestamp.now(),
            "method": "RMS_period_dection",
        }

        cc_periods = IntervalLabel(
            name = "cc_periods",
            time_index = periods, 
            time_start = t_ref, 
            metadata = metadata,
            plot_type = "box",
            plotstyle = DEFAULT_PLOT_STYLE.get("cc_periods", None)
        )

        ACC_channel.attach_label(cc_periods)

    def predict_circulation(self) -> None:
        """Predicts the circulation of a case by using the channels
        'cpr_acceleration' channel and the 'ecg_pads' channel.

        The procedure that is used has been published by Kern et al. in `10.1109/TBME.2023.3242717 <https://doi.org/10.1109/TBME.2023.3242717>`_.
        Here 'rosc_decision_function' is the output of the kernelized SVM used in and trained for the paper.

        Returns
        -------
        None.
        Adds three labels 'rosc_prediction', 'rosc_probability', and
        'rosc_decision_function'. 'rosc_decision_function' is a
        pseudo-probability computed from the decision function, and
        'rosc_prediction' is the binary prediction.
        """

        if (
            not (
                "cpr_acceleration" in self.get_channel_names()
                and "ecg_pads" in self.get_channel_names()
            )
            or "cc_depth_cont" in self.get_channel_names()
        ):
            logger.error(
                "WARNING! No Feedback-Sensor-Acceleration or ECG found. Check the presence of these channels in the case."
            )
        else:
            ACC_channel = self.data.get_channel("cpr_acceleration")
            acctime, acc = ACC_channel.get_data()  # get data

            ECG_channel = self.data.get_channel("ecg_pads")
            ecgtime, ecg = ECG_channel.get_data()  # get data

            if "cc_periods" not in self.get_label_names():
                self.find_CC_periods_acc()
 
            label_cc_periods = self.data.get_label("cc_periods")
            cc_periods, *_ = label_cc_periods.get_data()
            cc_periods =  np.asarray([t for pair in cc_periods for t in pair])

            t_ref = ACC_channel.time_start
            
            if label_cc_periods.is_time_relative():
                cc_periods = cc_periods.astype("timedelta64[s]").astype(float)
            else:
                cc_periods = np.array([(t - t_ref).total_seconds() for t in cc_periods])
            
            if ACC_channel.is_time_relative():
                acctime = acctime.astype("timedelta64[s]").astype(float)
            else:
                acctime = np.array([(t - t_ref).total_seconds() for t in acctime])

            # Time conversion for ECG channel
            if ECG_channel.is_time_relative():
                ecgtime = ecgtime.astype("timedelta64[s]").astype(float)
            else:
                ecgtime = np.array([(t - t_ref).total_seconds() for t in ecgtime])
            
            CC_starts = cc_periods[0::2]
            CC_stops = cc_periods[1::2]

            snippets = construct_snippets(
                acctime, acc, ecgtime, ecg, CC_starts, CC_stops
            )
            case_pred = predict_circulation(snippets)

            metadata = {
                "creator": "automatic",
                "creation_date": pd.Timestamp.now(),
                "method": "Period_dection",
            }


            pred_lab = Label(
                "rosc_prediction",
                case_pred["Starttime"],
                case_pred["Predicted"],
                time_start=t_ref,
                metadata=metadata,
                plotstyle=DEFAULT_PLOT_STYLE.get("rosc_prediction", None),
            )
            prob_lab = Label(
                "rosc_probability",
                case_pred["Starttime"],
                case_pred["Probability"],
                time_start=t_ref,
                metadata=metadata,
                plotstyle=DEFAULT_PLOT_STYLE.get("rosc_probability", None),
            )
            dec_lab = Label(
                "rosc_decision_function",
                case_pred["Starttime"],
                case_pred["DecisionFunction"],
                time_start=t_ref,
                metadata=metadata,
                plotstyle=DEFAULT_PLOT_STYLE.get("rosc_decision_function", None),
            )
            for lab in [pred_lab, prob_lab, dec_lab]:
                self.data.add_global_label(lab)


    def area_under_threshold(
        self,
        name: str,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
        threshold: int = 0
    ) -> ThresholdMetrics:
        """Calculates the area and duration where the signal falls
        below a specified threshold.
        
        The calculations might be used with a mean arterial pressure to asses for hypotension.
        They are implemented following the proposed metrics by Maheswari et al.
        `10.1213/ANE.0000000000003482 <https://doi.org/10.1213/ANE.0000000000003482>`_.

        See also
        --------
        
        :func:`.utils.helpers.area_under_threshold`

        Parameters
        ----------
        name
            The name of the label or channel - retrieved by meth:`.get_channel_or_label`.
            Allowed to be passed either as a positional or a keyword argument.
        start_time
            Start time for truncating the timeseries (meth:`truncate).
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

        index, data = self.get_channel_or_label(name).get_data()
        timeseries = pd.Series(data, index=index)

        return helpers.area_under_threshold(
            timeseries=timeseries,
            start_time=start_time,
            stop_time=stop_time,
            threshold=threshold
        )
