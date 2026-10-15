"""Common helpers for data import."""

from __future__ import annotations

import pandas as pd


def rename_channels(dats, new_name_dict):
    old_keys = list(dats.keys()).copy()
    for key in old_keys:
        if key not in ["Keys", "Main data", "Load Log"] and key in new_name_dict:
            if isinstance(dats[key], pd.DataFrame):
                dats[new_name_dict[key]] = dats[key].rename(
                    {key: new_name_dict[key]}, axis=1
                )
            else:
                dats[new_name_dict[key]] = dats[key].rename({key: new_name_dict[key]})

            if key != new_name_dict[key]:
                dats.pop(key)
    return dats
