"""Prediction of circulation (ROSC) during CPR."""

from __future__ import annotations

import logging


import numpy as np

from vitabel.analysis.circulation.features import data_filter
from vitabel.analysis.circulation.model import _load_circulation_model


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
    svc, scaler = _load_circulation_model()
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
