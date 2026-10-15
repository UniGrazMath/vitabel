"""Prediction of circulation (ROSC) during CPR."""

from __future__ import annotations

import logging


import numpy as np
import pandas as pd

from vitabel.analysis.circulation.features import construct_snippets, data_filter
from vitabel.analysis.circulation.model import load_circulation_model
from vitabel.timeseries import Channel, IntervalLabel, Label
from vitabel.utils.stylesheet import DEFAULT_PLOT_STYLE


logger = logging.getLogger("vitabel")


def predict_circulation(erg):
    acc_features = [
        "Acc Amp",
        "Pure Acc Corr",
        "Spectral Entropy ACC",
        "Mean ACC Power",
        "Mean ACC Kurtosis",
        "Mean ACC Skewness",
        "Mean ACC Median",
        "Mean ACC PeaktoPeak",
        "Mean ACC PP-RMS-Ratio",
        "PSD_03_Band",
        "PSD_36_Band",
        "PSD_69_Band",
        "PSD_912_Band",
        "PSD_1215_Band",
        "PSD_1518_Band",
        "PSD_Mean",
        "PSD_Std",
        "PSD_Kurt",
        "PSD_Skew",
        "PSD_Max1",
        "PSD_Maxfreq1",
        "Spectral OV",
        "Acc(ECG) Corr",
        "d2 AccCorr",
        "Arhy Corr",
        "Arhy Corr Quotient",
    ]
    ecg_features = [
        "ECG Amp",
        "Pure ECG Corr",
        "Power LEA",
        "Min ECG Length",
        "bS",
        "number Peaks",
        "Fibrillation Power",
        "High Freq Power",
        "Arhy Corr EKG",
        "RR_Mean",
        "RR_Std",
        "QRS_PP_Mean",
        "QRS_PP_Std",
        "QRS_width_mean",
        "QRS_width_std",
        "QRS_heigth_width_ratio",
        "ECG_slope_mean",
        "ECG_slop_Std",
        "ECG_slop_Kurtosis",
        "AMSA",
        "ECG_Fib_power",
        "ECG_norm_median",
        "ECG_norm_var",
    ]  # ,'Hjorth_mob','Hjorth_comp','ECG_skew'] #'Spectral Entropy ECG',
    all_features = acc_features + ecg_features
    train_keys = all_features

    background_keys = ["Start Time", "Max ACC", "Max ECG", "ACC ratio", "ECG ratio"]

    # Construct data arrays X and X_background for all features and all necessary background information, as well as label vector y
    N_total = len(erg["Snippet"]["Start Time"])
    n_f = len(train_keys)
    X = np.empty((N_total, n_f))
    y = np.empty(N_total)
    X_background = np.empty((N_total, len(background_keys) + 1))

    for i in range(N_total):
        k = 0
        for key in train_keys:
            X[i, k] = erg["Analysis"][key][i]
            k += 1
        X_background[i, -1] = i
        y[i] = 0
        k = 0
        for key in background_keys:
            X_background[i, k] = erg["Snippet"][key][i]
            k += 1
    y = y.astype(int)

    # Prefilter data
    Xn, Xn_background, yn = data_filter(
        X, X_background, y, background_keys=background_keys
    )
    # print(f"X {X.shape}")
    # print(f"Xn {Xn.shape}")

    # Load model and scaler
    svc, scaler = load_circulation_model()
    # Apply scaler
    Xn = scaler.transform(Xn)
    X = scaler.transform(X)

    # Predict results
    y_pred = svc.predict(X)  # Xnn
    dec = svc.decision_function(X)
    y_proba = svc.predict_proba(X)

    # Save results in array
    case_pred = {}
    case_pred = {
        "Predicted": np.array([]),
        "Real": np.array([]),
        "Probability": np.array([]),
        "DecisionFunction": np.array([]),
        "Starttime": np.array([]),
        "Index": np.array([]),
    }
    for i in Xn_background.T[-1].astype(
        int
    ):  # np.append(X_background.T[-1].astype(int),X_background.T[-1].astype(int)):
        case_pred["Predicted"] = np.append(case_pred["Predicted"], y_pred[i])
        case_pred["Real"] = np.append(case_pred["Real"], y[i])
        case_pred["Probability"] = np.append(case_pred["Probability"], y_proba[i][-1])
        case_pred["DecisionFunction"] = np.append(case_pred["DecisionFunction"], dec[i])
        case_pred["Starttime"] = np.append(
            case_pred["Starttime"], erg["Snippet"]["Start Time"][i]
        )
        case_pred["Index"] = np.append(case_pred["Index"], i)
        for j, key in enumerate(train_keys):
            if key not in case_pred:
                case_pred[key] = np.array([X[i, j]])
            else:
                case_pred[key] = np.append(case_pred[key], X[i, j])

        for j, key in enumerate(background_keys):
            if key not in case_pred:
                case_pred[key] = np.array([X_background[i, j]])
            else:
                case_pred[key] = np.append(case_pred[key], X_background[i, j])

    return case_pred


def find_ROSC_2(rosctime, roscdata, CC_starts, CC_stops):
    pred_time = rosctime
    pred = roscdata

    arrests = [CC_starts[0]]
    roscs = []

    i = 0
    analysis_interval_length = 20000
    pause_thresh = 60000
    CC_min_length = 10000
    final_flag = False
    while i < len(CC_stops) - 1 and not final_flag:
        while CC_starts[i + 1] - CC_stops[i] < pause_thresh and not final_flag:
            i += 1
            if i == len(CC_stops) - 1:
                final_flag = True
                break
        if final_flag:
            analysis_interval = [CC_stops[-1], CC_stops[-1] + analysis_interval_length]
        else:
            analysis_interval = [CC_stops[i], CC_stops[i] + analysis_interval_length]
        prob = np.mean(
            pred[
                (pred_time >= analysis_interval[0]) & (pred_time < analysis_interval[1])
            ]
        )
        while np.isnan(prob) and analysis_interval[1] - analysis_interval[0] < 120000:
            analysis_interval[1] += 5000
            prob = np.mean(
                pred[
                    (pred_time >= analysis_interval[0])
                    & (pred_time < analysis_interval[1])
                ]
            )

        if prob > 0.4:
            roscs.append(CC_stops[i])
            i += 1
            if not final_flag:
                while (CC_stops[i] - CC_starts[i]) < CC_min_length and not final_flag:
                    if i == len(CC_stops) - 1:
                        final_flag = True
                        break
                    i += 1
                if not final_flag:
                    arrests.append(CC_starts[i])
        else:
            if i == len(CC_stops) - 1:
                final_flag = True
            i += 1
    return roscs, arrests


def predict_rosc(
    cpr_acceleration_source: Channel,
    ecg_pads_source: Channel,
    label_cc_periods: IntervalLabel,
) -> list[Label]:
    """Predict circulation (ROSC) during CPR from the accelerometer signal of a
    CPR feedback sensor and the ECG signal of the defibrillator pads, as
    published by Kern et al. in :cite:`10.1109/tbme.2023.3242717`.

    Parameters
    ----------
    cpr_acceleration_source
        The channel containing the accelerometer signal of the CPR feedback sensor.
    ecg_pads_source
        The channel containing the ECG signal from the pads.
    label_cc_periods
        The periods of chest compressions (e.g., from
        :func:`.analysis.cpr.compressions.cc_periods_from_acceleration`).

    Returns
    -------
    The (unattached) labels ``rosc_prediction`` (binary prediction),
    ``rosc_probability``, and ``rosc_decision_function`` (output of the
    kernelized SVM trained for the paper).
    """
    acc_data = cpr_acceleration_source.get_data()  # get data
    acctime, acc = acc_data.time_index, acc_data.data

    ecg_data = ecg_pads_source.get_data()  # get data
    ecgtime, ecg = ecg_data.time_index, ecg_data.data

    cc_periods = label_cc_periods.get_data().time_index
    cc_periods = np.asarray([t for pair in cc_periods for t in pair])

    t_ref = cpr_acceleration_source.time_start

    if label_cc_periods.is_time_relative():
        cc_periods = (
            pd.Series(pd.to_timedelta(cc_periods)).dt.total_seconds().to_numpy()
        )
    else:
        cc_periods = np.array([(t - t_ref).total_seconds() for t in cc_periods])

    if cpr_acceleration_source.is_time_relative():
        acctime = acctime.dt.total_seconds().to_numpy()
    else:
        acctime = np.array([(t - t_ref).total_seconds() for t in acctime])

    # Time conversion for ECG channel
    if ecg_pads_source.is_time_relative():
        ecgtime = ecgtime.dt.total_seconds().to_numpy()
    else:
        ecgtime = np.array([(t - t_ref).total_seconds() for t in ecgtime])

    CC_starts = cc_periods[0::2]
    CC_stops = cc_periods[1::2]

    snippets = construct_snippets(acctime, acc, ecgtime, ecg, CC_starts, CC_stops)
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
    return [pred_lab, prob_lab, dec_lab]
