"""Analysis of defibrillations."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from vitabel.timeseries import Channel

logger = logging.getLogger("vitabel")


def extract_shocks(channels: list[Channel]) -> pd.DataFrame | None:
    """Collect all defibrillations with all stored information in a DataFrame.

    Defibrillation data is taken from the channels whose name contains
    ``"defibrillations"`` (e.g., ``defibrillations_DeliveredEnergy``); the
    columns of the resulting DataFrame are the channel names without the
    ``defibrillations_`` prefix, the index are the defibrillation times.

    Parameters
    ----------
    channels
        The channels to search for defibrillation data.

    Returns
    -------
    The DataFrame (empty if there is no defibrillation data), or ``None``
    if the defibrillation channels have different time indices.
    """
    defib_data = {}
    defib_channel_names = [
        channel_name
        for channel_name in [channel.name for channel in channels]
        if "defibrillations" in channel_name
    ]

    for channel in channels:
        if channel.name in defib_channel_names:
            channel_data = channel.get_data()
            time, data = channel_data.time_index, channel_data.data
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
