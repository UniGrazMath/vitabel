import json
import os


import numpy as np
import pandas as pd
import scipy.signal as sgn
import logging
import vitaldb

from typing import Any, TypeAlias, Union

from IPython.display import display
from pathlib import Path

from vitabel.timeseries import Channel, Label, IntervalLabel, TimeDataCollection
from vitabel.utils import (
    loading,
    constants,
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


Timedelta: TypeAlias = pd.Timedelta | np.timedelta64
Timestamp: TypeAlias = pd.Timestamp | np.datetime64

ChannelSpecification: TypeAlias = Union[str, dict[str, Any], "Channel"]
LabelSpecification: TypeAlias = Union[str, dict[str, Any], "Label"]

logger = logging.getLogger("vitabel")


class Vitals:
    """Container for vital data and labels.

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
        self.metadata: dict = {
            "Time_of_construction": str(
                pd.Timestamp.now()
            ),  # Contains background data, hashes and file names of the object
            "Recording_files_added": [],
            "Saving_to_json_time": "",
        }
        self.start_time = 0  # Start Time for the entire case
        self.channel_path = ""  # path where the channels are stored
        self.label_path = ""  # path where the labels are stored

    @property
    def channels(self) -> list[Channel]:
        return self.data.channels

    @property
    def labels(self) -> list[Label]:
        return self.data.labels

    def print_data_summary(self):
        """Print a summary of the data contained in the internal data collection."""
        self.data.print_summary()

    # -------------------------------------------------------------------------------------------
    # ---------------------- ADD RECORDINGS AND DATA --------------------------------------------
    # -------------------------------------------------------------------------------------------

    # Add a recording stored in filepath.
    def add_defibrillator_recording(self, filepath: Path | str, metadata={}):
        """Add the (defibrillator) recording to the cardio object.

        Vitabel allows to import the following defibrillator files:

            * ZOLL X-Series: Data needs to be exported as JSON or XML.
            * ZOLL E-Series and ZOLL AED-Pro: Data needs to be exported in two
              files as ``*_ecg.txt`` and ``*.xml``, where ``*`` is a wildcard string
              for the filename. filepath should be the path to the ``*_ecg.txt``
              file, and the xml-file should be placed in the same directory.
            * Stryker LIFEPAK 15: Data needs to be exported to XML in Stryker's
              CodeStat Software. At least the files ``*_Continuous.xml``,
              ``*_Continuous_Waveform.xml``, and ``*_CprEventLog.xml`` need to
              be present, such that a file can be loaded.
              The filepath should be the path to the ``*_Continuous.xml`` file.
              The other files should be in the same directory.
            * Stryker LUCAS: Data needs to be exported to XML in Stryker's CodeStat
              Software. At least the files ``*_Lucas.xml`` and
              ``*_CprEventLog.xml`` should be present.
              The filepath should be the path to the ``*_Lucas.xml`` file.
            * Corpuls: The Data needs to be exported as BDF file in Corpuls analysis
              software. During export the waveform data is stored in a ``*.bdf`` file,
              while the remaining events are exported in a directory with various files.
              The filepath should point to the ``*.bdf`` file. The Event-directory
              should be in the same directory as the ``*.bdf`` file.

        Parameters
        ----------
        filepath : pathlib.Path or str
            The path to the defibrillator recording. The string is parsed to pathlib.Path to be used as a Pathlib object
            The methods for loading the specific files are in the loading.py -Module
        Returns
        -------
        None.

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

                logger.info(f"File: {filename }.xml" "successfully read!")
        elif ".txt" in fileend_c:
            pure_filename = filename[: filename.index("_ecg")]

            pat_dat, dats = loading.read_zollcsv(
                dirpath.joinpath(Path(pure_filename + "_ecg.txt")),
                dirpath.joinpath(Path(pure_filename + ".xml")),
            )
            dats = rename_channels(dats, constants.zoll2channelnames_dict)

            logger.info(f"File: {filename } successfully read!")
        elif ".bdf" in fileend_c:
            pat_dat, dats = loading.read_corpuls(
                dirpath.joinpath(Path(filename + ".bdf"))
            )
            dats = rename_channels(dats, constants.corpuls2channelnames_dict)

            logger.info(f"File: {filename } successfully read!")

        elif fileend_c != []:
            logger.error(f"Error: No method to read { fileend_c } files!")
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
        vital_filepath: Path | str,
        metadata={"source": "VitalDB-Recording"},
    ):
        """Loading channels from a vitalDB recording.

        Parameters
        ----------
        vital_filepath : Path | str
            The path to the recording. Must be a ``*.vit`` file.

        Returns
        -------
        None.
        """
        vit = vitaldb.VitalFile(str(vital_filepath))
        df = vit.to_pandas(vit.get_track_names(), interval=None, return_datetime=True)
        df.set_index("Time", inplace=True, drop=True)
        self.add_data_from_DataFrame(df, metadata=metadata)

    def add_old_cardio_label(self, file_path: Path | str):
        """Add labels from old "cardio"-version of this code.

        Can read both consensual as well as singular annotations.

        Parameters
        ----------
        file_path : Path | str
            Path to old cardio_label.

        Returns
        -------
        None.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation {file_path} not found.")

        if file_path.suffix == ".json":
            label_dict = {}
            with open(file_path, "r") as fp:  # Load Annotations
                ann_dict = json.load(fp)
            for label in ann_dict["Merged"]:
                if label != "Time":
                    label_dict[label] = {
                        "timestamp": ann_dict["Merged"][label],
                        "data": None,
                    }
            for label in ann_dict["One-Annotator"]:
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
                compression_channel = self.data.get_channel("CC")
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
        elif file_path.suffix == ".csv":
            anno = pd.read_csv(file_path)
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
        source: dict[str, any],
        name: str,
        metadata: dict = {},
        time_start=None,
        datatype: str = "channel",
        anchored_channel: Channel | None = None,
    ):
        """Adds a channel or label from a dict containing a single timeseries.

        The dict needs to have two keys: 'timestamp' and 'data'.

        Parameters
        ----------
        source : dict[str, array]
            Contains the data in the from {'timestamp': [], 'data' : []}
        name : str
            The name of the channel.
        metadata : dict, optional
            Metadata for the timeseries. The default is {}.
        time_start : TYPE, optional
            time_start value for the timeseries, in case of a relative timeseries. The default is None.
        datatype : str, optional
            Either 'channel' or 'label' or 'interval_label' depending on which kind of data to attach. The default is "channel".
        anchored_channel :  Channel | None
            In case of datatype = 'label', where to attach the label. None means global label. The default is None

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
        else:
            if time_start:
                time_start = pd.Timestamp(time_start)

            time = source["timestamp"]
            data = source["data"]
            if len(time) > 0:
                if datatype == "channel":
                    cha = Channel(
                        name,
                        time,
                        data,
                        metadata=metadata,
                        time_start=time_start,
                    )
                    self.data.add_channel(cha)
                elif datatype == "label" and anchored_channel is None:
                    cha = Label(
                        name,
                        time,
                        data,
                        metadata=metadata,
                        time_start=time_start,
                    )
                    self.data.add_global_label(cha)
                elif datatype == "label" and anchored_channel is not None:
                    cha = Label(
                        name,
                        time,
                        data,
                        metadata=metadata,
                        time_start=time_start,
                        anchored_channel=anchored_channel,
                    )
                elif datatype == "interval_label" and anchored_channel is None:
                    cha = IntervalLabel(
                        name,
                        time,
                        data,
                        metadata=metadata,
                        time_start=time_start,
                    )
                    self.data.add_global_label(cha)

    def add_data_from_dict(
        self,
        source: dict[str, dict] | Path,
        metadata: dict = {},
        time_start=None,
        datatype: str = "channel",
        anchored_channel: Channel | None = None,
    ):
        """Add multiple channels from a dict.

                Each value must be a dict accepted by  '_add_single_dict(value, key)'

                Parameters
                ----------
                source : dict[str, dict] | Path
                    The data which is added in the from
                    ``{'key1': {'timestamp' : [], 'data' : []}, 'key2': {...} ,... }}}``.
                    If source is a Path, then it is loaded via ``json.load(source)``.
                metadata : dict, optional
                    Metadata applicable to all timeseries. The default is ``{}``.
                time_start : TYPE, optional
                    time_start value for the timeseries, in case of a relative timeseries.
                    The default is None.
                datatype : str, optional
                    Either ``'channel'`` or ``'label'`` or ``'interval_label'`` depending on
                    which kind of labels to attach.. The default is 'channel'.
                anchored_channel :  Channel | None
                    In case of datatype = ``'label'``, where to attach the label. None means
                    global label. The default is ``None``.


                Raises
                ------
                ValueError
                    In case the dictionary does not has the expected form.
        .

                Returns
                -------
                None.

        """
        if isinstance(source, Path):
            with open(source, "r") as file:
                source = json.load(file)

        for key in source:
            if not isinstance(source[key], dict):
                raise ValueError(
                    f" Source must be a dictionary of the form {{'channel1': {{'timestamp':[...]', 'data':[] }}, 'channel2':{{...}} }}.  For key {key} the value is not a dict."
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
        metadata={},
        time_start: str | None = None,
        time_unit=None,
        datatyp="channel",
        anchored_channel: Channel | None = None,
    ):
        """Adds Data from a pandas.DataFrame.

        Parameters
        ----------
        source : pandas.DataFrame
            The DataFrame containing the data. The index of the DataFrame contains the
            time (either as DatetimeIndex or numeric Index),
            and the columns contain the channels. NaN-Values in the columns are
            not taken into account an ignored.
        metadata : dict, optional
            A dictionary containing all the metadata for the channels/labels.
            Is parsed to channel/Label and saved there as general argument.
        time_start : pd.Timestamp() or str or None, optional
            A starting time for the data. Must be accepted by pd.Timestamp(time_start)
            In case the index is numeric. The times will be interpreted as relative
            to this value. The default is 0 and means no information is given.
        datatype : str, optional
            Either 'channel' or 'label' or 'interval_label' depending on which kind
            of labels to attach. The default is "channel".
        anchored_channel :  Channel | None
            In case of datatype = 'label', where to attach the label. None means
            global label. The default is None

        Raises
        ------
        ValueError
            The DataFrame does not contain a DateTime or Numeric Index.

        Returns
        -------
        None.

        """

        if not (
            isinstance(source.index, pd.DatetimeIndex)
            or (pd.api.types.is_numeric_dtype(source.index))
        ):
            raise ValueError(
                "The DataFrame needs to have a datetime or a numeric index, "
                "which describes the time of the timeseries."
            )
        else:
            for col in source.columns:
                series = source[col]
                series = series[series.notna()]
                time = np.array(series.index)
                data = series.values
                if len(time) > 0:
                    if datatyp == "channel":
                        cha = Channel(
                            name=col,
                            time_index=time,
                            data=data,
                            time_start=time_start,
                            time_unit=time_unit,
                            metadata=metadata,
                        )
                        self.data.add_channel(cha)
                    elif datatyp == "label" and anchored_channel is None:
                        cha = Label(
                            col,
                            time,
                            data,
                            time_start=time_start,
                            time_unit=time_unit,
                            metadata=metadata,
                        )
                        self.data.add_global_label(cha)
                    elif datatyp == "label" and anchored_channel is not None:
                        cha = Label(
                            col,
                            time,
                            data,
                            time_start=time_start,
                            time_unit=time_unit,
                            metadata=metadata,
                            anchored_channel=anchored_channel,
                        )

    def add_data_from_csv(
        self,
        file_path: Path | str,
        time_start=None,
        time_unit=None,
        metadata={},
        **kwargs,
    ):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        df = pd.read_csv(file_path, **kwargs)
        self.add_data_from_DataFrame(
            df, time_start=time_start, time_unit=time_unit, metadata=metadata
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
        **kwargs
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
        **kwargs
            The plot style properties to set. Passing ``None``
            unsets the key from the plotstyle dictionary.
        """
        self.data.set_label_plotstyle(label_specification, **kwargs)

    def save_data(self, path: Path | str):
        """
        Saves the channels to a JSON-File. Saves also channel_path and a hash in the metadata dict.

        Parameters
        ----------
        path : str, optional
            The path of the JSON File. This is parsed to pathlib.Path. If not provided, then the self.channel_path attribute is used. If this is not set, a Value Error is raised.
        Returns
        -------
        None.

        """

        if isinstance(path, str):
            path = Path(path)

        self.metadata["Saving_to_json_time"] = str(pd.Timestamp.now())
        self.metadata["filepath"] = str(path)
        has = self.data.channel_data_hash()
        data_dict = self.data.to_dict()
        json_dict = {"metadata": self.metadata, "data": data_dict, "hash": has}

        # Write down saving
        with open(path, "w") as fd:
            json.dump(json_dict, fd, cls=NumpyEncoder)

    def load_data(self, path: Path | str, check_channel_hash=True):
        """Load the channel_information from path.

        Parameters
        ----------
        path : str
            The Path of the channel data. It is parsed to pathlib.Path.

        Returns
        -------
        None.

        """

        path = Path(path)
        with open(path, "r") as fd:
            json_dict = json.load(fd)
        self.metadata = json_dict["metadata"]
        self.data = TimeDataCollection.from_dict(json_dict["data"])
        saved_hash = json_dict["hash"]
        channel_hash = self.data.channel_data_hash()

        if check_channel_hash:
            saved_hash = channel_hash
            if channel_hash != saved_hash:
                raise ValueError(
                    "Saved hash value  is not equal to hash of  reloaded channels. Use keyword-argument 'check_channel_hash = False' to ignore conflicting hash values."
                )

            # Add Offsets from metadata labels the channel json contains the raw data, without offset correction

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Retrieve data -------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def get_channels(self, name: str | None = None, **kwargs) -> list[Channel]:
        return self.data.get_channels(name, **kwargs)

    def get_channel(self, name: str | None = None, **kwargs) -> Channel:
        return self.data.get_channel(name, **kwargs)

    def get_labels(self, name: str | None = None, **kwargs) -> list[Label]:
        return self.data.get_labels(name, **kwargs)

    def get_label(self, name: str | None = None, **kwargs) -> Label:
        return self.data.get_label(name, **kwargs)

    def get_channels_or_labels(
        self, name: str | None = None, **kwargs
    ) -> list[Channel | Label]:
        return self.data.get_channels(name, **kwargs) + self.data.get_labels(
            name, **kwargs
        )

    def get_channel_or_label(
        self, name: str | None = None, **kwargs
    ) -> Channel | Label:
        channels_or_labels = self.get_channels_or_labels(name, **kwargs)
        if len(channels_or_labels) != 1:
            raise ValueError(
                "Channel or Label specification was ambiguous, no unique channel or Label "
                f"was identified. Query returned: {channels_or_labels}"
            )
        return channels_or_labels[0]

    def get_channel_names(self):  # part of register application
        """Returns a list with the names of all channels.

        Returns
        -------
        channel_names : list
            List of all channel names.

        """
        return self.data.channel_names

    def get_label_names(self):  # part of register application
        """Returns a list with the names of all labels.

        Returns
        -------
        label_names : list
            List of all label names.

        """
        return self.data.label_names

    def get_channel_or_label_names(self):
        """
        # Returns a list with the names of all channels and labels.

        Returns
        -------
        data_names : list
            List of all channel and label names.

        """

        return self.get_channel_names() + self.get_label_names()

    def keys(self):
        return self.get_channel_or_label_names()

    def rec_start(self):  # part of register application
        """Returns the start_time value of a recording as timestamp.

        Returns
        -------
        pd.Timestamp
            The reference time_value of a recording.

        """
        if self.data.is_time_absolute():
            start_time = self.data.channels[0].time_start
            for chan in self.channels:
                if chan.time_start < start_time:
                    start_time = chan.time_start

        return start_time

    def rec_stop(self):  # part of register application
        """Returns the start_time value of a recording as timestamp.

        Returns
        -------
        pd.Timestamp
            The reference time_value of a recording.

        """
        if self.data.is_time_absolute():
            stop_time = (
                self.data.channels[0].time_start + self.data.channels[0].time_index[-1]
            )
            for chan in self.channels:
                cha_stop_time = chan.time_start + chan.time_index[-1]
                if cha_stop_time > stop_time:
                    stop_time = cha_stop_time

        return stop_time

    def get_channel_infos(self):
        """Returns information about all channels.

        Parameters
        ----------
        dataframe : bool, optional
            Whether result is formated as dict or DataFrame. The default is True.

        Returns
        -------
        info_dict : pd.DataFrame or dict
            Information DataFrame about all channels.

        """
        info_dict = {}
        for i, chan in enumerate(self.data.channels):
            name = chan.name
            chan_time, chan_data = chan.get_data()

            info_dict[i] = {}
            info_dict[i]["Name"] = name
            for key in chan.metadata:
                info_dict[i][key] = chan.metadata[key]
            if len(chan_time) > 0:
                for key, value in zip(
                    ["first_entry", "last_entry", "length", "offset"],
                    [np.min(chan_time), np.max(chan_time), len(chan_time), chan.offset],
                ):
                    info_dict[i][key] = value
            else:
                for key, value in zip(
                    ["first_entry", "last_entry", "length", "offset"],
                    [None, None, 0, chan.offset],
                ):
                    info_dict[i][key] = value

        info_dict = pd.DataFrame(info_dict).transpose()
        return info_dict

    def get_label_infos(self):
        """Returns information about all labels.

        Parameters
        ----------.
        dataframe : bool, optional
            Whether result is formated as dict or DataFrame. The default is True.

        Returns
        -------
        info_dict : pd.DataFrame or dict
            Information DataFrame about all channels.

        """
        info_dict = {}
        for i, chan in enumerate(self.data.labels):
            name = chan.name
            chan_time, chan_data = chan.get_data()

            info_dict[i] = {}
            info_dict[i]["Name"] = name
            for key in chan.metadata:
                info_dict[i][key] = chan.metadata[key]
            if len(chan_time) > 0:
                for key, value in zip(
                    ["first_entry", "last_entry", "length", "offset"],
                    [np.min(chan_time), np.max(chan_time), len(chan_time), chan.offset],
                ):
                    info_dict[i][key] = value
            else:
                for key, value in zip(
                    ["first_entry", "last_entry", "length", "offset"],
                    [None, None, 0, chan.offset],
                ):
                    info_dict[i][key] = value
        info_dict = pd.DataFrame(info_dict).transpose()
        return info_dict

    def info(self):
        """Displays relevant information about the channels and labels
        currently present in the recording.

        Returns
        -------
        None.

        """
        channel_info = self.get_channel_infos()
        label_info = self.get_label_infos()
        display(channel_info)
        display(label_info)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Plotting tools ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------

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
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        return self.data.plot_interactive(
            channels=channels,
            labels=labels,
            start=start,
            stop=stop,
            time_unit=time_unit,
            include_attached_labels=include_attached_labels,
            channel_overviews=channel_overviews,
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
        mode="threshold",
        breaththresh: float = 4,
        etco2_thresh: float = 4,
    ):
        """Computes the etCO2 values and the timestamps of the ventilations based
        on CO2 waveform and add it to the labels.

        The capnography signal must be a channel of the file and needs to be named 'capnography'. Currently two different versions are implemented: 'threshold' and 'filter'. The default is 'filter'.

        Parameters
        ----------
        mode : str, optional,
            Which method to use to detect ventilations from CO2 signal. Either 'filter' which is a unpublished method by Wolfgang Kern or 'threshold',
            which is the method presented by  Aramendi et al. "Feasibility of the capnogram to monitor ventilation rate during cardiopulmonary resuscitation" DOI: 10.1016/j.resuscitation.2016.08.033
        breaththresh : float, optional
            Threshold value below which a minimum is identified as ventilation/respiration . The default is 2 (mmHg).
        etco2_thresh : float, optional
            Threshold value above which a maximum is identified as etCo2 value of an expiration. The default is 4 (mmHg).


        """
        if "capnography" not in self.get_channel_names():
            logger.error(
                "Error! No Capnography Signal found. Cannot compute etCO2 and detect ventilations"
            )
        else:
            co2_channel = self.data.get_channel("capnography")
            cotime, co = co2_channel.get_data()  # get data

            freq = np.timedelta64(1, "s") / np.nanmedian(cotime.diff())
            cotime = np.asarray(cotime)
            co = np.asarray(co)
            if mode == "filter":
                but = sgn.butter(4, 1 * 2 / freq, btype="lowpass", output="sos")
                co2 = sgn.sosfiltfilt(but, co)  # Filter forwarsd and backward
                et_index = sgn.find_peaks(co2, distance=1 * freq, height=etco2_thresh)[
                    0
                ]  # find peaks of filtered signal as markers for etco2
                resp_index = sgn.find_peaks(
                    -co2, distance=1 * freq, height=-breaththresh
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

                # id there is no maximum
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

            elif mode == "threshold":
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

    def cycle_duration_analysis(self):
        """Determine start and end of periods of continuous chest compressions
        based on single chest compression markers.

        Adds two labels ``cc_period_start`` and ``cc_period_stop`` to the data set.

        .. SEEALSO::

            The method is described in DOI: 10.1016/j.resuscitation.2021.12.028 or in the
            Thesis 'Towards a data-driven cardiac arrest treatment' by Wolfgang Kern in more detail.
            See `https://unipub.uni-graz.at/obvugrhs/content/titleinfo/10138095`__ for more information.
        """
        if (
            "cc" not in self.data.channel_names
            and "cc_depth" not in self.data.channel_names
        ):
            logger.error(
                "Case contains no compression markers. Cycle duration analysis can not be computed."
            )
            return

        if "cc" in self.data.channel_names:
            CC_channel = self.data.get_channel("cc")
        else:
            CC_channel = self.data.get_channel("cc_depth")
        comp, data = CC_channel.get_data()  # get data
        comp = np.sort(comp)
        if CC_channel.is_time_relative():
            comp = np.asarray([pd.Timedelta(c).total_seconds() for c in comp])
        else:
            comp = np.asarray(
                [pd.Timedelta(c - CC_channel.time_start).total_seconds() for c in comp]
            )

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
            "method": "Cycle_duration_analysis",
        }

        if CC_channel.is_time_absolute():
            time_start = CC_channel.time_start
        else:
            time_start = None

        sta_lab = Label(
            "cc_period_start",
            time_index=sta,
            data=None,
            time_start=time_start,
            metadata=metadata,
            plotstyle=DEFAULT_PLOT_STYLE.get("cc_period_start", None),
        )
        CC_channel.attach_label(sta_lab)
        sto_lab = Label(
            "cc_period_stop",
            time_index=sto,
            data=None,
            time_start=time_start,
            metadata=metadata,
            plotstyle=DEFAULT_PLOT_STYLE.get("cc_period_stop", None),
        )
        CC_channel.attach_label(sto_lab)

    def find_CC_periods_acc(self):  # part of register application
        """Determines start and stop of periods with continuous chest compressions.

        The procedure is implemented as described in
        DOI: 10.1016/j.resuscitation.2021.12.028 and DOI: 10.1016/j.dib.2022.107973

        Requires a channel 'cpr_acceleration' in the recording, which is the signal of an accelerometry-based feedback sensor for cardiopulmonary resuscitation.

        Returns
        -------
        None.
        Adds two labels 'cc_period_start_acc' and 'cc_period_stop_acc' to the recording.

        """
        if "cpr_acceleration" not in self.get_channel_names():
            logger.error(
                "No Acceleration data found. Can not identify CC-periods via acceleration."
            )
            return

        ACC_channel = self.data.get_channel("cpr_acceleration")
        acctime, acc = ACC_channel.get_data()  # get data
        freq = np.timedelta64(1, "s") / np.nanmedian(acctime.diff())
        if ACC_channel.is_time_relative():
            acctime = np.asarray([pd.Timedelta(c).total_seconds() for c in acctime])
        else:
            acctime = np.asarray(
                [
                    pd.Timedelta(c - ACC_channel.time_start).total_seconds()
                    for c in acctime
                ]
            )
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

        metadata = {
            "creator": "automatic",
            "creation_date": pd.Timestamp.now(),
            "method": "Period_dection",
        }
        if ACC_channel.is_time_absolute():
            time_start = ACC_channel.time_start
        else:
            time_start = None

        sta_lab = Label(
            "cc_period_start_acc",
            starts,
            None,
            time_start=time_start,
            metadata=metadata,
            plotstyle=DEFAULT_PLOT_STYLE.get("cc_period_start_acc", None),
        )
        ACC_channel.attach_label(sta_lab)
        sto_lab = Label(
            "cc_period_stop_acc",
            stops,
            None,
            time_start=time_start,
            metadata=metadata,
            plotstyle=DEFAULT_PLOT_STYLE.get("cc_period_stop_acc", None),
        )
        ACC_channel.attach_label(sto_lab)

    def predict_circulation(self):
        """Predicts the circulation of a case by using the channels
        'cpr_acceleration' channel and the 'ecg_pads' channel.

        The procedure that is used has been published by Kern et al. in
        DOI: 10.1109/TBME.2023.3242717.

        Adds three labels 'rosc_prediction', 'rosc_probability', and
        'rosc_decision_function'. Here 'rosc_decision_function' is the output
        of the kernelized SVM used in the paper, 'rosc_decision_function' is a
        pseudo-probability computed from the decision function, and
        'rosc_prediction' is the binary prediction.

        Returns
        -------
        None.

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

            if ("cc_period_start_acc" not in self.get_label_names()) and (
                "cc_period_stop_acc" not in self.get_label_names()
            ):
                self.find_CC_periods_acc()

            CC_period_start_label = self.data.get_label("cc_period_start_acc")
            CC_starts, data = CC_period_start_label.get_data()  # get data

            CC_period_stop_label = self.data.get_label("cc_period_stop_acc")
            CC_stops, data = CC_period_stop_label.get_data()  # get data

            if ACC_channel.is_time_relative():
                acctime = np.asarray([pd.Timedelta(c).total_seconds() for c in acctime])
                CC_starts = np.asarray(
                    [pd.Timedelta(c).total_seconds() for c in CC_starts]
                )
                CC_stops = np.asarray(
                    [pd.Timedelta(c).total_seconds() for c in CC_stops]
                )

            else:
                acctime = np.asarray(
                    [
                        pd.Timedelta(c - ACC_channel.time_start).total_seconds()
                        for c in acctime
                    ]
                )
                CC_starts = np.asarray(
                    [
                        pd.Timedelta(c - ACC_channel.time_start).total_seconds()
                        for c in CC_starts
                    ]
                )
                CC_stops = np.asarray(
                    [
                        pd.Timedelta(c - ACC_channel.time_start).total_seconds()
                        for c in CC_stops
                    ]
                )

            if ECG_channel.is_time_relative():
                ecgtime = np.asarray([pd.Timedelta(c).total_seconds() for c in ecgtime])
            else:
                ecgtime = np.asarray(
                    [
                        pd.Timedelta(c - ACC_channel.time_start).total_seconds()
                        for c in ecgtime
                    ]
                )

            snippets = construct_snippets(
                acctime, acc, ecgtime, ecg, CC_starts, CC_stops
            )
            case_pred = predict_circulation(snippets)

            metadata = {
                "creator": "automatic",
                "creation_date": pd.Timestamp.now(),
                "method": "Period_dection",
            }
            if ACC_channel.is_time_absolute():
                time_start = ACC_channel.time_start
            else:
                time_start = None

            pred_lab = Label(
                "rosc_prediction",
                case_pred["Starttime"],
                case_pred["Predicted"],
                time_start=time_start,
                metadata=metadata,
                plotstyle=DEFAULT_PLOT_STYLE.get("rosc_prediction", None),
            )
            prob_lab = Label(
                "rosc_probability",
                case_pred["Starttime"],
                case_pred["Probability"],
                time_start=time_start,
                metadata=metadata,
                plotstyle=DEFAULT_PLOT_STYLE.get("rosc_probability", None),
            )
            dec_lab = Label(
                "rosc_decision_function",
                case_pred["Starttime"],
                case_pred["DecisionFunction"],
                time_start=time_start,
                metadata=metadata,
                plotstyle=DEFAULT_PLOT_STYLE.get("rosc_decision_function", None),
            )
            for lab in [pred_lab, prob_lab, dec_lab]:
                self.data.add_global_label(lab)
