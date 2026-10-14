import bz2
import copy
import shutil
import tarfile

import pytest
import numpy as np
import numpy.typing as npt

from pathlib import Path

from vitabel import Vitals


TEST_DATA_DIR: Path = Path(__file__).parent / "data"
EXAMPLE_DATA_DIR: Path = Path(__file__).parent.parent / "examples" / "data"

TIME_2ECG: npt.NDArray = np.load(TEST_DATA_DIR / "time_2ecg.npy")


@pytest.fixture
def data_time_2ecg():
    yield TIME_2ECG.copy()


@pytest.fixture
def vitabel_test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture
def vitabel_example_data_dir():
    return EXAMPLE_DATA_DIR


@pytest.fixture
def fixed_random_seed():
    np.random.seed(42)
    yield
    np.random.seed(None)


def _decompress_bz2(source: Path, target_dir: Path) -> Path:
    """Decompress ``source`` (a ``*.bz2`` file) into ``target_dir``."""
    target = target_dir / source.with_suffix("").name
    with bz2.open(source, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


# ---------------------------------------------------------------------------
# Session-scoped test data. Decompressing / extracting (and loading) the
# larger recordings only once per test session keeps the test suite fast.
#
# The ``*_vitals`` fixtures are shared between tests and must be treated as
# read-only. Tests that modify a recording need to work on a copy, e.g.
# via the corresponding ``*_vitals_copy`` fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def zoll_json_path(tmp_path_factory) -> Path:
    """Path to the decompressed ZOLL X Series example recording (JSON)."""
    target_dir = tmp_path_factory.mktemp("zoll_json")
    return _decompress_bz2(EXAMPLE_DATA_DIR / "ZOLL_test_case.json.bz2", target_dir)


@pytest.fixture(scope="session")
def lifepak_dir(tmp_path_factory) -> Path:
    """Directory containing the extracted LIFEPAK 15 test recording
    (``LP15_Continuous.xml``, ``LP15_Continuous_Waveform.xml``,
    ``LP15_CprEventLog.xml``).
    """
    target_dir = tmp_path_factory.mktemp("lifepak")
    with tarfile.open(TEST_DATA_DIR / "LP15.tar.gz", "r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(target_dir, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            archive.extractall(target_dir)
    return target_dir


@pytest.fixture(scope="session")
def edf_sample_path(tmp_path_factory) -> Path:
    """Path to the decompressed EDF+ sample recording."""
    target_dir = tmp_path_factory.mktemp("edf")
    return _decompress_bz2(
        TEST_DATA_DIR / "sample_signals" / "test_generator_2.edf.bz2", target_dir
    )


@pytest.fixture(scope="session")
def zoll_vitals(zoll_json_path) -> Vitals:
    """The ZOLL example recording, loaded once per session. Read-only!"""
    recording = Vitals()
    recording.add_defibrillator_recording(zoll_json_path)
    return recording


@pytest.fixture
def zoll_vitals_copy(zoll_vitals) -> Vitals:
    """A modifiable copy of the ZOLL example recording."""
    return copy.deepcopy(zoll_vitals)


@pytest.fixture(scope="session")
def lifepak_vitals(lifepak_dir) -> Vitals:
    """The LIFEPAK 15 test recording, loaded once per session. Read-only!"""
    recording = Vitals()
    recording.add_defibrillator_recording(lifepak_dir / "LP15_Continuous.xml")
    return recording
