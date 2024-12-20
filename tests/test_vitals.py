import bz2
import shutil
import tempfile
import pandas as pd
import numpy as np
import json
import pytest

from pathlib import Path

from vitabel import Vitals
from vitabel import Channel, Label


def compare_two_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    else:
        for key in dict1:
            if type(dict1[key]) is not type(dict2[key]):
                return False
            else:
                if isinstance(dict1[key], list):
                    return all(dict1[key] == dict2[key])
                elif isinstance(dict1[key], np.ndarray):
                    return (dict1[key] == dict2[key]).all()
                elif isinstance(dict1[key], dict):
                    return compare_two_dictionaries(dict1, dict2)
                else:
                    return dict1[key] == dict2[key]


def test_cardio_init():
    empty_cardio_object = Vitals()

    assert len(empty_cardio_object.channels) == 0
    assert len(empty_cardio_object.labels) == 0


def test_add_defibrillator_file_without_suffix(vitabel_test_data_dir):
    lifepak_dir = vitabel_test_data_dir
    file_path_no_suffix = lifepak_dir / "ZOLL_test_case"
    lifepak_recording = Vitals()

    with pytest.raises(
        FileNotFoundError, match="File not found in directory. Check path!"
    ):
        lifepak_recording.add_defibrillator_recording(file_path_no_suffix)


# DATA NOT AVAILABLE FOR PUBLICATION
# def test_add_defibrillator_lifepak15(vitabel_test_data_dir):
#     lifepak_dir = vitabel_test_data_dir / "lifepak"
#     file_path = lifepak_dir / "2022112410042400-LP158890Schwein9-Hup-Sup-ROSC_Continuous.xml"
#     lifepak_recording = Vitals()
#     lifepak_recording.add_defibrillator_recording(file_path)
#     assert len(lifepak_recording.channels)>0
#     check_cardio_properties(lifepak_recording)

# def test_add_incomplete_defibrillator_lifepak15(vitabel_test_data_dir):
#     lifepak_dir = vitabel_test_data_dir / "lifepak"
#     file_path = lifepak_dir / "2022112310110800-LP158890Schwein8-Hup-Sup-ROSC_Continuous.xml"
#     with pytest.raises(
#         FileNotFoundError, match="Error when Loading LIFEPAK Recording! Expected Files"
#     ):
#         lifepak_recording = Vitals()
#         lifepak_recording.add_defibrillator_recording(file_path)

# def test_add_defibrillator_lifepak15_2(vitabel_test_data_dir):
#     lifepak_dir = vitabel_test_data_dir / "lifepak"
#     file_path = lifepak_dir / "5_6390640_Continuous.xml"
#     lifepak_recording = Vitals()
#     lifepak_recording.add_defibrillator_recording(file_path)
#     assert len(lifepak_recording.channels)>0
#     check_cardio_properties(lifepak_recording)


def test_add_zoll_json(vitabel_example_data_dir):
    cardio_recording = Vitals()
    compressed_defi_file = vitabel_example_data_dir / "ZOLL_test_case.json.bz2"
    with tempfile.TemporaryDirectory() as tmpdir:
        defi_file = Path(tmpdir) / compressed_defi_file.stem
        with bz2.open(compressed_defi_file, "rb") as source:
            with open(defi_file, "wb") as dest:
                shutil.copyfileobj(source, dest)
        cardio_recording.add_defibrillator_recording(defi_file)

    assert sorted(cardio_recording.get_channel_names()) == [
        "capnography",
        "cc_depth",
        "cc_rate",
        "cc_release_velocity",
        "cpr_acceleration",
        "defibrillations_Current",
        "defibrillations_DefaultEnergy",
        "defibrillations_DeliveredEnergy",
        "defibrillations_Impedance",
        "defibrillations_Nr",
        "ecg_filtered",
        "ecg_pads",
        "etco2",
        "heart_rate",
        "impedance",
        "mean_inspired_co2",
        "nibp_dia",
        "nibp_map",
        "nibp_sys",
        "ppg",
        "respiratory_rate",
        "spo2",
        "temperature_1",
    ]
    assert all(not channel.is_empty() for channel in cardio_recording.channels)


# DATA NOT AVAILABLE FOR PUBLICATION

# def test_add_zoll_xml(vitabel_test_data_dir):
#     file_path = vitabel_test_data_dir / "20140720135017_AR12I001978.xml"
#     cardio_recording = Vitals()
#     cardio_recording.add_defibrillator_recording(file_path)
#     check_cardio_properties(cardio_recording)
#     assert cardio_recording.channels


# def test_add_zoll_csv(vitabel_test_data_dir):
#     file_path = vitabel_test_data_dir / "_20110825193454_00012254_ecg.txt"
#     cardio_recording = Vitals()
#     cardio_recording.add_defibrillator_recording(file_path)
#     check_cardio_properties(cardio_recording)
#     assert cardio_recording.channels


def test_add_stryker_lucas(vitabel_test_data_dir):
    lifepak_dir = vitabel_test_data_dir
    file_path = lifepak_dir / "Lucas_file_Lucas.xml"
    lifepak_recording = Vitals()
    lifepak_recording.add_defibrillator_recording(file_path)
    assert lifepak_recording.channels


def test_add_empty_stryker_lucas(vitabel_test_data_dir):
    lifepak_recording = Vitals()
    lifepak_recording.add_defibrillator_recording(
        vitabel_test_data_dir / "empty_lucas_Lucas.xml"
    )
    assert not lifepak_recording.channels


def test_add_incomplete_stryker_lucas(vitabel_test_data_dir):
    lifepak_dir = vitabel_test_data_dir
    file_path = lifepak_dir / "incomplete_lucas_Lucas.xml"
    with pytest.raises(
        FileNotFoundError, match="Error when Loading LUCAS Recording! Expected Files"
    ):
        lifepak_recording = Vitals()
        lifepak_recording.add_defibrillator_recording(file_path)


# DATA NOT AVAILABLE FOR PUBLICATION
# def test_add_corpuls_1(vitabel_test_data_dir):
#     corpuls_dir = vitabel_test_data_dir  / "Corpuls_Case_1"
#     file_path = corpuls_dir / "146-AED-18321258-00000000.bdf"
#     cardio_recording = Vitals()
#     cardio_recording.add_defibrillator_recording(file_path)
#     assert cardio_recording.channels


# def test_add_corpuls_2(vitabel_test_data_dir):
#     corpuls_dir = vitabel_test_data_dir / "Corpuls_Case_2"
#     file_path = corpuls_dir / "20000101000646-13000015EA9D8601-00000000.bdf"
#     cardio_recording = Vitals()
#     cardio_recording.add_defibrillator_recording(file_path)
#     assert cardio_recording.channels

# def test_add_corpuls_2_no_additional_event_data(vitabel_test_data_dir):
#     corpuls_dir = vitabel_test_data_dir / "Corpuls_Case_3"
#     file_path = corpuls_dir / "20000101000646-13000015EA9D8601-00000000.bdf"
#     cardio_recording = Vitals()
#     cardio_recording.add_defibrillator_recording(file_path)
#     assert cardio_recording.channels


def test_add_vital_db_recording(vitabel_test_data_dir):
    recording_path = vitabel_test_data_dir / "vital_file.vit"
    cardio_object = Vitals()
    cardio_object.add_vital_db_recording(recording_path)
    assert cardio_object.channels


def test_add_unreadable_file(vitabel_test_data_dir):
    recording_path = vitabel_test_data_dir / "test_stylesheet_demo_plot.png"
    cardio_object = Vitals()
    cardio_object.add_defibrillator_recording(recording_path)
    assert not cardio_object.channels


def test_add_nonexistent_file(vitabel_test_data_dir):
    recording_path = vitabel_test_data_dir / "nonexistent.png"
    cardio_object = Vitals()
    with pytest.raises(
        FileNotFoundError, match="File not found in directory. Check path!"
    ):
        cardio_object.add_defibrillator_recording(recording_path)


# Comptability Function to old cardio version: No test required.
# def test_add_old_json_labels(vitabel_test_data_dir):
#     recording_path = vitabel_test_data_dir / "359_AR15J015538-20200103-111629-3088_annf.json"
#     cardio_object = Vitals()
#     CC_channel = Channel('CC',[pd.Timestamp(2020,11,24,1,0,0) , pd.Timestamp(2020,11,24,1,0,1)],None)
#     cardio_object.add_channel(CC_channel)
#     cardio_object.add_old_cardio_label(recording_path)
#     assert cardio_object.labels

# def test_add_old_csv_labels(vitabel_test_data_dir):
#     recording_path = vitabel_test_data_dir / "221124_so_ann.csv"
#     cardio_object = Vitals()
#     cardio_object.add_old_cardio_label(recording_path)
#     assert cardio_object.labels
#     for key in ['Case', 'Annotator', 'Time', 'Duration /s']:
#         assert key in cardio_object.labels[0].metadata

# def test_add_nonexistant_old_label(vitabel_test_data_dir):
#     recording_path = vitabel_test_data_dir / "nonexistant.csv"
#     cardio_object = Vitals()
#     with pytest.raises(
#         FileNotFoundError, match="Annotation"
#     ):
#         cardio_object.add_old_cardio_label(recording_path)

# def test_add_unreadable_old_label(vitabel_test_data_dir):
#     recording_path = vitabel_test_data_dir / "test_stylesheet_demo_plot.png"
#     cardio_object = Vitals()
#     with pytest.raises(
#         ValueError, match="is not a valid cardio 1.x annotation"
#     ):
#         cardio_object.add_old_cardio_label(recording_path)


def test_add_data_from_dict():
    data_dct = {
        "timestamp": [
            pd.Timestamp(2022, 11, 24, 1, 0, 0),
            pd.Timestamp(2022, 11, 24, 1, 0, 1),
        ],
        "data": [0, 0],
    }
    dct = {"channel_1": data_dct}
    cardio_object = Vitals()
    cardio_object.add_data_from_dict(dct)
    cardio_object.add_data_from_dict(dct, datatype="label")
    assert cardio_object.channels


def test_add_data_from_incorrect_dict():
    data_dct = {
        "wrong_time_name": [
            pd.Timestamp(2022, 11, 24, 1, 0, 0),
            pd.Timestamp(2022, 11, 24, 1, 0, 1),
        ],
        "data": [0, 0],
    }
    dct1 = {"channel_1": data_dct}
    with pytest.raises(
        ValueError, match="The dictionary must contain a 'timestamp' and a 'data' key "
    ):
        cardio_object = Vitals()
        cardio_object.add_data_from_dict(dct1)
    data_dct = {
        "timestamp": [
            pd.Timestamp(2022, 11, 24, 1, 0, 0),
            pd.Timestamp(2022, 11, 24, 1, 0, 1),
        ],
        "wrong_data_name": [0, 0],
    }
    dct2 = {"channel_2": data_dct}
    with pytest.raises(
        ValueError, match="The dictionary must contain a 'timestamp' and a 'data' key "
    ):
        cardio_object = Vitals()
        cardio_object.add_data_from_dict(dct2)

    dct3 = {"channel_3": [1, 1, 1]}
    with pytest.raises(ValueError, match="Source must be a dictionary of the form "):
        cardio_object = Vitals()
        cardio_object.add_data_from_dict(dct3)


def test_add_data_from_incorrect_DataFrame():
    df = pd.DataFrame.from_dict(
        {
            "timestamp": [
                pd.Timestamp(2022, 11, 24, 1, 0, 0),
                pd.Timestamp(2022, 11, 24, 1, 0, 1),
            ],
            "data": [0, 0],
            "strings": ["a", "b"],
        }
    )
    df.set_index("strings", inplace=True)
    cardio_object = Vitals()
    with pytest.raises(
        ValueError,
        match="The DataFrame needs to have a datetime or a numeric index, which describes the time of the timeseries.",
    ):
        cardio_object.add_data_from_DataFrame(df)
    with pytest.raises(
        ValueError,
        match="The DataFrame needs to have a datetime or a numeric index, which describes the time of the timeseries.",
    ):
        cardio_object.add_data_from_DataFrame(df, datatyp="label")


def test_add_label_data_from_DataFrame():
    df = pd.DataFrame.from_dict(
        {
            "timestamp": [
                pd.Timestamp(2022, 11, 24, 1, 0, 0),
                pd.Timestamp(2022, 11, 24, 1, 0, 1),
            ],
            "data": [0, 0],
        }
    )
    df.set_index("timestamp", inplace=True)
    cardio_object = Vitals()
    cardio_object.add_data_from_DataFrame(df, datatyp="label")
    assert cardio_object.labels


def test_add_csv_data(vitabel_example_data_dir):
    recording_path = vitabel_example_data_dir / "capno.csv.bz2"
    cardio_object = Vitals()
    cardio_object.add_data_from_csv(
        recording_path,
        time_start=pd.Timestamp(1970, 1, 1, 0, 0, 0),
        time_unit="ms",
        metadata={"source": "volucapno"},
        index_col="Timestamp",
    )
    assert cardio_object.channels


def test_add_channel():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    assert "Channel1" in cardio_object.get_channel_names()


def test_remove_channel():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cardio_object.remove_channel(channel=cha)
    assert "Channel1" not in cardio_object.get_channel_names()


def test_add_global_label():
    cha = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_global_label(cha)
    assert "Label1" in cardio_object.get_label_names()


def test_remove_global_label():
    cha = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_global_label(cha)
    cardio_object.remove_label(label=cha)
    assert "Label1" not in cardio_object.get_label_names()


def test_get_channels():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cha2 = cardio_object.get_channels("Channel1")[0]
    assert cha == cha2


def test_get_labels():
    cha = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_global_label(cha)
    cha2 = cardio_object.get_labels("Label1")[0]
    assert cha == cha2


def test_get_channel():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cha2 = cardio_object.get_channel("Channel1")
    assert cha == cha2


def test_get_label():
    cha = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_global_label(cha)
    cha2 = cardio_object.get_label("Label1")
    assert cha == cha2


def test_get_data_names():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    lab = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cardio_object.add_global_label(lab)
    assert cardio_object.get_channel_or_label_names() == ["Channel1", "Label1"]
    assert cardio_object.keys() == ["Channel1", "Label1"]


def test_rec_start():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cha2 = Channel(
        "Channel2",
        [
            "2020-01-01 00:00:01.000000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cardio_object.add_channel(cha2)
    t_start = cardio_object.rec_start()
    assert t_start == pd.Timestamp("2020-01-01 00:00:01.000000")


def test_get_channel_info():
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
        metadata={"test": "1"},
    )
    cha2 = Channel(
        "Channel2",
        [
            "2020-01-01 00:00:01.000000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cardio_object.add_channel(cha2)
    info_dict = cardio_object.get_channel_infos()
    assert isinstance(info_dict, pd.DataFrame)
    assert "Channel2" in np.asarray(info_dict["Name"])
    assert "test" in info_dict.columns
    assert 1 in info_dict["test"]


def test_get_label_info():
    cha = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
        metadata={"test": "1"},
    )
    cha2 = Label(
        "Label2",
        [
            "2020-01-01 00:00:01.000000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cardio_object.add_channel(cha2)
    info_dict = cardio_object.get_channel_infos()
    assert isinstance(info_dict, pd.DataFrame)
    assert "Label2" in np.asarray(info_dict["Name"])
    assert "test" in info_dict.columns
    assert 1 in info_dict["test"]


def test_saving_and_loading(tmpdir):
    cha = Channel(
        "Channel1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
    )
    lab = Label(
        "Label1",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([1, 2, 3]),
        metadata={"test": "testtext"},
    )
    cardio_object = Vitals()
    cardio_object.add_channel(cha)
    cardio_object.add_global_label(lab)
    filepath = tmpdir / "testdata.json"
    cardio_object.save_data(filepath)

    cardio_object2 = Vitals()
    cardio_object2.load_data(filepath)
    assert cardio_object.data == cardio_object2.data


def test_create_shock_information_DataFrame():
    shock_energie = Channel(
        "defibrillations_default_energy",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([175.9, 174.8, 174.0]),
    )
    shock_impedance = Channel(
        "defibrillations_impedance",
        [
            "2020-04-13 02:48:16.666000",
            "2020-04-13 02:50:34.445000",
            "2020-04-13 02:56:57.449000",
        ],
        np.array([69.1, 68.9, 70.3]),
    )
    cardio_object = Vitals()
    cardio_object.channels.append(shock_energie)
    cardio_object.channels.append(shock_impedance)
    shocks = cardio_object.shocks()
    assert isinstance(shocks, pd.DataFrame)
    assert isinstance(shocks.index, pd.DatetimeIndex)
    assert "default_energy" in shocks.columns
    assert "impedance" in shocks.columns


def test_etco2_and_ventilation_detection(vitabel_test_data_dir):
    co_df = pd.read_csv(
        vitabel_test_data_dir / "sample_signals" / "Capnography.csv.bz2"
    )
    t = np.asarray(co_df["Time / s"])
    data = np.asarray(co_df["CO2 mmHg, Waveform"])
    data = data[t > 2855]
    t = t[t > 2855]
    cardio_object = Vitals()
    co2_channel = Channel(
        "capnography", t, data, time_start=pd.Timestamp(2024, 1, 1, 0, 0, 0)
    )
    cardio_object.add_channel(co2_channel)
    cardio_object.compute_etco2_and_ventilations()
    vent_dict = cardio_object.get_label("ventilations_from_capnography").to_dict()
    etCO2_dict = cardio_object.get_label("etco2_from_capnography").to_dict()

    assert cardio_object.labels
    assert "ventilations_from_capnography" in cardio_object.get_label_names()
    assert "etco2_from_capnography" in cardio_object.get_label_names()

    np.testing.assert_allclose(
        vent_dict["time_index"][:10],
        [0.0, 1.504, 3.752, 16.2, 24.408, 26.776, 31.192, 33.504, 37.264, 41.808],
    )
    np.testing.assert_allclose(
        etCO2_dict["time_index"][:10],
        [0.0, 1.496, 11.408, 16.144, 24.4, 26.728, 32.584, 35.568, 39.912, 44.24],
    )
    np.testing.assert_allclose(
        etCO2_dict["data"][:10],
        [
            22.06545738,
            22.03992386,
            24.04979742,
            15.04962875,
            23.09704632,
            11.72470137,
            16.67250414,
            29.04956834,
            28.05089976,
            25.04570468,
        ],
    )


def test_etco2_and_ventilation_detection_threshold_mode(vitabel_test_data_dir):
    file_path = vitabel_test_data_dir / "sample_signals" / "Capnography.csv.bz2"
    co_df = pd.read_csv(file_path)
    t = np.asarray(co_df["Time / s"])
    data = np.asarray(co_df["CO2 mmHg, Waveform"])
    data = data[t > 2855]
    t = t[t > 2855]
    cardio_object = Vitals()
    co2_channel = Channel(
        "capnography", t, data, time_start=pd.Timestamp(2024, 1, 1, 0, 0, 0)
    )
    cardio_object.channels.append(co2_channel)
    cardio_object.compute_etco2_and_ventilations(mode="threshold")
    assert cardio_object.labels
    assert "ventilations_from_capnography" in cardio_object.get_label_names()
    assert "etco2_from_capnography" in cardio_object.get_label_names()


def test_etco2_and_ventilation_detection_relative_time(vitabel_test_data_dir):
    file_path = vitabel_test_data_dir / "sample_signals" / "Capnography.csv.bz2"
    co_df = pd.read_csv(file_path)
    t = co_df["Time / s"]
    data = co_df["CO2 mmHg, Waveform"]
    cardio_object = Vitals()
    co2_channel = Channel("capnography", t, data)
    cardio_object.channels.append(co2_channel)
    cardio_object.compute_etco2_and_ventilations()
    assert cardio_object.labels
    assert "ventilations_from_capnography" in cardio_object.get_label_names()
    assert "etco2_from_capnography" in cardio_object.get_label_names()


def test_cycle_duration_analysis_absolute(vitabel_test_data_dir):
    len_pause_range = [0, 10, 10, 5, 1]
    n_compressions_range = np.asarray([30, 20, 2, 20, 20])
    compressions = np.array([0])
    dt = 0.6
    for n_compressions, len_pause in zip(n_compressions_range, len_pause_range):
        new_compressions = np.linspace(
            compressions[-1] + len_pause,
            compressions[-1] + len_pause + n_compressions * dt,
            n_compressions,
        )
        compressions = np.append(compressions, new_compressions)

    cardio_object = Vitals()
    CC_channel = Channel(
        "cc", compressions, time_start=pd.Timestamp(2024, 1, 1, 0, 0, 0)
    )
    cardio_object.channels.append(CC_channel)
    cardio_object.cycle_duration_analysis()

    CC_Start_dict = cardio_object.get_label("cc_period_start").to_dict()
    CC_Stop_dict = cardio_object.get_label("cc_period_stop").to_dict()

    with open(vitabel_test_data_dir / "CC_Start_test_data.json", "r") as fd:
        CC_Start_dict_test = json.load(fd)
    with open(vitabel_test_data_dir / "CC_Stop_test_data.json", "r") as fd:
        CC_Stop_dict_test = json.load(fd)
    assert (CC_Start_dict["time_index"] == CC_Start_dict_test["time_index"]).all()
    assert (CC_Stop_dict["time_index"] == CC_Stop_dict_test["time_index"]).all()


def test_cycle_duration_analysis_relative():
    len_pause_range = [0, 10, 10, 5, 1]
    n_compressions_range = np.asarray([30, 20, 2, 20, 20])
    compressions = np.array([0])
    dt = 0.6
    for n_compressions, len_pause in zip(n_compressions_range, len_pause_range):
        new_compressions = np.linspace(
            compressions[-1] + len_pause,
            compressions[-1] + len_pause + n_compressions * dt,
            n_compressions,
        )
        compressions = np.append(compressions, new_compressions)

    cardio_object = Vitals()
    CC_channel = Channel("cc", compressions)
    cardio_object.channels.append(CC_channel)
    cardio_object.cycle_duration_analysis()
    assert cardio_object.labels
    assert "cc_period_start" in cardio_object.get_label_names()
    assert "cc_period_stop" in cardio_object.get_label_names()


def test_acceleration_CC_period_dection_absolute(vitabel_test_data_dir):
    file_path = vitabel_test_data_dir / "sample_signals" / "Accelerometer.csv.bz2"
    co_df = pd.read_csv(file_path)
    t = co_df["Time / s"]
    data = co_df["CPR Acceleration"]
    cardio_object = Vitals()
    acc_channel = Channel(
        "cpr_acceleration", t, data, time_start=pd.Timestamp(2024, 1, 1, 0, 0, 0)
    )
    cardio_object.channels.append(acc_channel)
    cardio_object.find_CC_periods_acc()
    CC_Start_dict = cardio_object.get_label("cc_period_start_acc").to_dict()
    CC_Stop_dict = cardio_object.get_label("cc_period_stop_acc").to_dict()
    with open(vitabel_test_data_dir / "CC_Start_acc_test_data.json", "r") as fd:
        CC_Start_dict_test = json.load(fd)
    with open(vitabel_test_data_dir / "CC_Stop_acc_test_data.json", "r") as fd:
        CC_Stop_dict_test = json.load(fd)
    assert (CC_Start_dict["time_index"] == CC_Start_dict_test["time_index"]).all()
    assert (CC_Stop_dict["time_index"] == CC_Stop_dict_test["time_index"]).all()


def test_acceleration_CC_period_dection_relative(vitabel_test_data_dir):
    file_path = vitabel_test_data_dir / "sample_signals" / "Accelerometer.csv.bz2"
    co_df = pd.read_csv(file_path)
    t = co_df["Time / s"]
    data = co_df["CPR Acceleration"]
    cardio_object = Vitals()
    acc_channel = Channel("cpr_acceleration", t, data)
    cardio_object.channels.append(acc_channel)
    cardio_object.find_CC_periods_acc()
    assert cardio_object.labels
    assert "cc_period_start_acc" in cardio_object.get_label_names()
    assert "cc_period_start_acc" in cardio_object.get_label_names()


def test_rosc_detection_absolute_time(vitabel_test_data_dir):
    file_path1 = vitabel_test_data_dir / "sample_signals" / "Accelerometer.csv.bz2"
    co_df = pd.read_csv(file_path1)
    acctime = co_df["Time / s"]
    acc = co_df["CPR Acceleration"]
    cardio_object = Vitals()
    acc_channel = Channel(
        "cpr_acceleration", acctime, acc, time_start=pd.Timestamp(2024, 1, 1, 0, 0, 0)
    )
    cardio_object.channels.append(acc_channel)

    file_path2 = vitabel_test_data_dir / "sample_signals" / "ShockElectrodes.csv.bz2"
    co_df = pd.read_csv(file_path2)
    ecgtime = co_df["Time / s"]
    ecg = co_df["Pads"]
    print(f"ECG-Mean  {np.mean(np.abs(ecg))}")
    ecg_channel = Channel(
        "ecg_pads", ecgtime, ecg, time_start=pd.Timestamp(2024, 1, 1, 0, 0, 0)
    )
    cardio_object.channels.append(ecg_channel)
    cardio_object.predict_circulation()
    assert cardio_object.labels


def test_rosc_detection_relative_time(vitabel_test_data_dir):
    file_path1 = vitabel_test_data_dir / "sample_signals" / "Accelerometer.csv.bz2"
    co_df = pd.read_csv(file_path1)
    acctime = co_df["Time / s"]
    acc = co_df["CPR Acceleration"]
    cardio_object = Vitals()
    acc_channel = Channel("cpr_acceleration", acctime, acc)
    cardio_object.channels.append(acc_channel)

    file_path2 = vitabel_test_data_dir / "sample_signals" / "ShockElectrodes.csv.bz2"
    co_df = pd.read_csv(file_path2)
    ecgtime = co_df["Time / s"]
    ecg = co_df["Pads"]
    print(f"ECG-Mean  {np.mean(np.abs(ecg))}")
    ecg_channel = Channel("ecg_pads", ecgtime, ecg)
    cardio_object.channels.append(ecg_channel)
    cardio_object.predict_circulation()
    assert cardio_object.labels


def test_real_evaluation(vitabel_example_data_dir):
    cardio_recording = Vitals()

    compressed_defi_file = vitabel_example_data_dir / "ZOLL_test_case.json.bz2"
    with tempfile.TemporaryDirectory() as tmpdir:
        defi_file = Path(tmpdir) / compressed_defi_file.stem
        with bz2.open(compressed_defi_file, "rb") as source:
            with open(defi_file, "wb") as dest:
                shutil.copyfileobj(source, dest)
        cardio_recording.add_defibrillator_recording(defi_file)

    cardio_recording.compute_etco2_and_ventilations()
    assert cardio_recording.labels
    assert "ventilations_from_capnography" in cardio_recording.get_label_names()
    assert "etco2_from_capnography" in cardio_recording.get_label_names()

    cardio_recording.cycle_duration_analysis()
    assert "cc_period_start" in cardio_recording.get_label_names()
    assert "cc_period_stop" in cardio_recording.get_label_names()
    cardio_recording.predict_circulation()


def test_analysis_exception_for_missing_data(caplog):
    cardio_recording = Vitals()
    cardio_recording.compute_etco2_and_ventilations()
    cardio_recording.find_CC_periods_acc()
    cardio_recording.cycle_duration_analysis()
    cardio_recording.predict_circulation()
    assert len(cardio_recording.labels) == 0
    assert all(
        warning in caplog.text
        for warning in [
            "No Capnography Signal found.",
            "No Acceleration data found.",
            "Case contains no compression markers.",
            "No Feedback-Sensor-Acceleration or ECG found.",
        ]
    )
