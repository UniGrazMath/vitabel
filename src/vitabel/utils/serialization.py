"""Serialization helpers (compressed array encoding, JSON encoding)."""

from __future__ import annotations

import base64
import datetime
import io
import json

import numpy as np
import numpy.typing as npt
import pandas as pd

from pathlib import Path

__all__ = ["compress_array", "decompress_array", "NumpyEncoder"]


def compress_array(lst: npt.NDArray | pd.DatetimeIndex | pd.TimedeltaIndex) -> str:
    """Compress a given array by returning a bae64-encoded string of
    the corresponding (NumPy) zip compressed data.
    """
    buff = io.BytesIO()
    np.savez_compressed(buff, data=lst)
    return base64.b64encode(buff.getvalue()).decode()


def decompress_array(b64data: str) -> npt.NDArray:
    """Decompress a given base64 string representing a
    (NumPy) zipped array object.
    """
    buff = io.BytesIO(base64.b64decode(b64data))
    return np.load(buff)["data"]


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype != object:
            return compress_array(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype == object:
            return obj.tolist()
        elif isinstance(obj, pd.TimedeltaIndex):
            return compress_array(obj)
        elif isinstance(obj, pd.DatetimeIndex):
            return compress_array(obj)
        elif isinstance(obj, datetime.time):
            return obj.strftime("%H:%M:%S")
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %X")
        elif isinstance(obj, datetime.datetime):
            return str(obj)
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
