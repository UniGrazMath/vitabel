import pytest
import numpy as np
import numpy.typing as npt

from pathlib import Path


TIME_2ECG: npt.NDArray = np.load(Path(__file__).parent / "data" / "time_2ecg.npy")


@pytest.fixture
def data_time_2ecg():
    yield TIME_2ECG.copy()


@pytest.fixture
def vitabel_test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def vitabel_example_data_dir():
    return Path(__file__).parent.parent / "examples" / "data"


@pytest.fixture
def fixed_random_seed():
    np.random.seed(42)
    yield
    np.random.seed(None)
