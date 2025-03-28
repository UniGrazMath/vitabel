__all__ = ["DEFAULT_PLOT_STYLE"]

DEFAULT_PLOT_STYLE = {
    "capnography": {
        "color": "goldenrod",
        "label": "Capnography",
        "alpha": 0.8,
    },
    "cpr_acceleration": {
        "color": "blue",
        "label": "Accelerometry",
        "alpha": 0.6,
    },
    "ecg_filtered": {
        "color": "darkgreen",
        "label": "Filtered ECG",
        "alpha": 0.6,
    },
    "ecg_i": {
        "color": "green",
        "label": "ECG I",
        "alpha": 0.6,
    },
    "ecg_ii": {
        "color": "green",
        "label": "ECG II",
        "alpha": 0.6,
    },
    "ecg_iii": {
        "color": "green",
        "label": "ECG III",
        "alpha": 0.6,
    },
    "ecg_pads": {
        "color": "green",
        "label": "ECG (Pads)",
        "alpha": 0.6,
    },
    "impedance": {
        "color": "red",
        "label": "Impedance",
        "alpha": 0.6,
    },
    "heart_rate": {
        "color": "forestgreen",
        "label": "Heart rate",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "*",
    },
    "mean_inspired_co2": {
        "color": "darkgoldenrod",
        "label": "Mean Inspired CO$_2$",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "s",
    },
    "etco2": {
        "color": "red",
        "label": "etCO$_2$",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "s",
        "mfc": "orangered",
    },
    "respiratory_rate": {
        "color": "darkgoldenrod",
        "label": "Respiratory Rate",
        "alpha": 0.6,
        "linestyle": ":",
        "marker": "s",
    },
    "cc_depth": {
        "color": "purple",
        "label": "Chest Compression Depth",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "o",
    },
    "cc_release_velocity": {
        "color": "purple",
        "label": "Chest Compression Release Velocity",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "o",
    },
    "cc_rate": {
        "color": "purple",
        "label": "Chest Compression Rate",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "o",
    },
    "ppg": {
        "color": "darkcyan",
        "label": "Photoplethysmography",
        "alpha": 0.6,
        "linestyle": "-",
        "marker": "",
    },
    "nibp_map": {
        "color": "red",
        "label": "Non-Invasive Mean Blood Pressure",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "d",
    },
    "nibp_sys": {
        "color": "red",
        "label": "Non-Invasive Systolic Blood Pressure",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "v",
    },
    "nibp_dia": {
        "color": "red",
        "label": "Non-Invasive Diastolic Blood Pressure",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "^",
    },
    "defibrillations_Nr": {
        "label": "Defibrillation",
        "alpha": 0.8,
        "linestyle": "",
        "marker": r"$\mapsdown$",
        "ms": 12,
        "markeredgecolor": "orangered",
        "markerfacecolor": "gold",
    },
    "defibrillations_DefaultEnergy": {
        "label": "Defibrillation",
        "alpha": 0.8,
        "linestyle": "",
        "marker": r"$\mapsdown$",
        "ms": 12,
        "markeredgecolor": "orangered",
        "markerfacecolor": "gold",
    },
    "defibrillations_DeliveredEnergy": {
        "label": "Defibrillation",
        "alpha": 0.8,
        "linestyle": "",
        "marker": r"$\mapsdown$",
        "ms": 12,
        "markeredgecolor": "orangered",
        "markerfacecolor": "gold",
    },
    "time_12_lead_ecg": {
        "label": "12-Lead ECG Analysis",
        "alpha": 0.8,
        "linestyle": "",
        "marker": r"$\sinewave$",
        "ms": 12,
        "markeredgecolor": "orangered",
        "markerfacecolor": "gold",
    },
    # "rosc_probability": {
    #     "color": "darkred",
    #     "label": "ROSC Probability",
    #     "alpha": 0.35,
    #     "linestyle": "",
    #     "marker": "o",
    # },
    # "etco2_from_capnography": {
    #     "color": "red",
    #     "label": "etCO$_2$ from Capnography Data",
    #     "alpha": 0.8,
    #     "linestyle": "",
    #     "marker": "s",
    #     "ms" : 6,
    # },
    # "ventilations_from_capnography": {
    #     "label": "Ventilation (Capnography)",
    #     "alpha": 0.6,
    #     "linestyle": "",
    #     "marker": "v",
    #     "ms": 10,
    #     "markeredgecolor": "goldenrod",
    #     "markerfacecolor": "gold" ,
    # },
    "rosc_probability": {
        "color": "darkred",
        "label": "ROSC Probability",
        "alpha": 0.35,
        "linestyle": "",
        "marker": "o",
    },
    "etco2_from_capnography": {
        "color": "darkred",
        "label": "etCO$_2$ from Capnography Data",
        "alpha": 0.8,
        "linestyle": "",
        "marker": "s",
        "ms": 2,
    },
    "ventilations_from_capnography": {
        "label": "Ventilation (Capnography)",
        "alpha": 0.6,
        "linestyle": "",
        "marker": "$V$",
        "ms": 12,
        "markeredgecolor": "goldenrod",
        "markerfacecolor": "gold",
    },
    "EKG": {
        "label": "ECG",
        "color": "green",
        "alpha": 0.6,
    },
    "PAP" : {
        "label" : "Pulmonary Artery",
        'linestyle' : '-' ,
        'marker' : '',
        'color' : '#FFD700',
    },
}
