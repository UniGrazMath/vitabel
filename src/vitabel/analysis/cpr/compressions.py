"""Analysis of chest compressions (CC periods, chest compression fraction)."""

from __future__ import annotations

import logging


import numpy as np


logger = logging.getLogger("vitabel")


def convert_two_alternating_list(df):
    lis = []
    for index, value in df.iterrows():
        lis.extend([index, float(value.iloc[0])])
    return lis


def CCF_minute(t_start, t_stop, CC_starts, CC_stops):
    CC_starts_min = CC_starts[(CC_starts >= t_start) & (CC_starts < t_stop)]
    CC_stops_min = CC_stops[(CC_stops >= t_start) & (CC_stops < t_stop)]
    if len(CC_starts_min) > 0 and len(CC_stops_min) > 0:
        if len(CC_starts_min) == len(CC_stops_min):
            if CC_stops_min[0] < CC_starts_min[0]:
                CC_starts_min = np.insert(CC_starts_min, 0, t_start)
                CC_stops_min = np.append(CC_stops_min, t_stop)
        elif len(CC_starts_min) > len(CC_stops_min):
            CC_stops_min = np.append(CC_stops_min, t_stop)
        else:
            CC_starts_min = np.insert(CC_starts_min, 0, t_start)

    elif len(CC_starts_min) == 0 and len(CC_stops_min) == 0:
        last_CC_start = CC_starts[CC_starts < t_start]
        last_CC_stop = CC_stops[CC_stops < t_start]
        if len(last_CC_start) == 0 and len(last_CC_stop) == 0:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_start])
        elif len(last_CC_start) != 0 and len(last_CC_stop) == 0:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_stop])
        elif last_CC_start[-1] < last_CC_stop[-1]:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_start])
        else:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_stop])

    elif len(CC_starts_min) > 0:
        CC_stops_min = np.array([t_stop])
    elif len(CC_stops_min) > 0:
        CC_starts_min = np.array([t_start])

    return np.sum(CC_stops_min - CC_starts_min) / 60000
