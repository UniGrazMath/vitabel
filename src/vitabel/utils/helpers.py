"""Collection of various auxiliary and helper functions."""

import antropy
import base64
import datetime
import io
import joblib
import json

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.fft as fft
import scipy.signal as sgn
import scipy.stats as stats

from pathlib import Path

__all__ = [
    "deriv",
    "integrate",
    "construct_snippets",
    "av_mean",
    "predict_circulation",
    "rename_channels",
    "NumpyEncoder",
    "determine_gaps_in_recording",
    "linear_interpolate_gaps_in_recording",
]


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
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def match_object(obj: object, **kwargs) -> bool:
    """Check whether a given object matches the filter criteria.

    Parameters
    ----------
    obj
        The object to check.
    kwargs
        Filter criteria. The keys are the attribute names of the object,
        and the values are the expected values. If expected value is a
        dictionary, then the object attribute only needs to contain all
        key-value pairs in the dictionary (but could contain other key-value
        pairs too).
    """
    for key, value in kwargs.items():
        if isinstance(value, dict):
            if not all(getattr(obj, key).get(k) == v for k, v in value.items()):
                return False
        else:
            if not hasattr(obj, key):
                return False
            if getattr(obj, key) != value:
                return False
    return True


def deriv(time, data):
    """
    Calculates the central point first derivative of x=time,y=data,
    and take forward/backward step derivative at the boundaries

    Parameters
    ----------
    time : np.array()
        equidistantly spaced x values.
    data : np.array()
        corresponding y values.

    Returns
    -------
    ddat : np.array()
        array with the derivative values.
    """
    h = time[2:] - time[:-2]
    if np.std(h) == 0:
        h = h[0] / 2
        ddat = data[2:] - data[:-2]
        ddat = ddat / (2 * h)
        ddat = np.insert(ddat, 0, (data[1] - data[0]) / h)
        ddat = np.append(ddat, (data[-1] - data[-2]) / h)
        return ddat
    else:
        ddat = data[2:] - data[0:-2]
        ddat = ddat / (h)
        ddat = np.insert(ddat, 0, (data[1] - data[0]) / (time[1] - time[0]))
        ddat = np.append(ddat, (data[-1] - data[-2]) / (time[-1] - time[-2]))
        return ddat


def integrate(time, data):
    """
    Integrates data after time with trapezoidal rule

    Parameters
    ----------
    time : pd.Series
        Timestamps in seconds.
    data : pd.Series
        Datapoints.

    Returns
    -------
    vel : pd.Series
        Integrated data. vel[0]=0

    """
    if len(time) != len(data):
        raise Exception("Length of time and data must agree")
    time = list(time)
    data = list(data)
    n = len(data)
    vel = [0]
    deltat = time[1] - time[0]
    for i in range(n - 1):
        meana = (data[i] + data[i + 1]) / 2
        vel.append(vel[i] + meana * deltat)
        i += 1
    return pd.Series(vel)


def construct_snippets(acctime, acc, ecgtime, ecg, CC_starts, CC_stops):
    sniplen = 1000
    features = [
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
    ]  # ,'Hjorth_mob','Hjorth_comp','ECG_skew']# LIST of all Features
    background_infos = [
        "ACC",
        "ECG",
        "Start Time",
        "Max ACC",
        "Max ECG",
        "Max CO2",
        "ACC ratio",
        "ECG ratio",
    ]
    snippets = {"Snippet": {}, "Type": [], "Analysis": {}}  # Contruct key
    snippets["Analysis"] = dict([(feature, []) for feature in features])
    snippets["Snippet"] = dict([(typ, []) for typ in background_infos])
    CC_starts = np.append(CC_starts, acctime[-1])

    k = 0
    for sto, sta in zip(CC_stops, CC_starts[1:]):
        n_snippets = (sta - sto) // 2 - 1
        r = (sta - sto) % 2
        i = 0
        while i < n_snippets:
            cond_a = (acctime >= sto + r + 2 * i) & (acctime < sto + r + 2 * (i + 2))
            cond_e = (ecgtime >= sto + r + 2 * i) & (ecgtime < sto + r + 2 * (i + 2))
            acc_snippet = acc[cond_a].astype(float)
            acctime_snippet = acctime[cond_a]
            ecg_snippet = ecg[cond_e].astype(float)
            # ecgtime_snippet = ecgtime[cond_e]  # TODO: was unused?
            if (len(acc_snippet) == sniplen) & (len(ecg_snippet) == sniplen):
                acc_snippet -= np.mean(acc_snippet)
                ecg_snippet -= np.mean(ecg_snippet)
                snippets["Snippet"]["ACC"].append(acc_snippet)
                snippets["Snippet"]["ECG"].append(ecg_snippet)

                snippets["Snippet"]["Start Time"].append(acctime_snippet[0])

                snippets["Snippet"]["Max ACC"].append(np.max(np.abs(acc_snippet)))
                snippets["Snippet"]["Max ECG"].append(np.max(np.abs(ecg_snippet)))
                snippets["Snippet"]["ACC ratio"].append(
                    np.max(np.abs(acc_snippet)) / np.mean(np.abs(acc_snippet))
                )
                snippets["Snippet"]["ECG ratio"].append(
                    np.max(np.abs(ecg_snippet)) / np.mean(np.abs(ecg_snippet))
                )

                (
                    sp_cor,
                    acc_amp,
                    ekg_amp,
                    z_max,
                    ze_max,
                    za_max,
                    d2z,
                    arhy_cor,
                    arhy_cor_ekg,
                    arhy_cor_quot,
                    spec_ent_acc,
                    ekg_feat,
                    ac_f,
                    ek_f,
                ) = ekg_acc_corr1(
                    np.arange(0, sniplen * 0.004, 0.004),
                    acc_snippet,
                    ecg_snippet,
                    nperse=sniplen,
                )

                snippets["Analysis"]["Spectral OV"].append(sp_cor)
                snippets["Analysis"]["Acc Amp"].append(acc_amp)
                snippets["Analysis"]["ECG Amp"].append(ekg_amp)
                snippets["Analysis"]["Pure ECG Corr"].append(ze_max)
                snippets["Analysis"]["Acc(ECG) Corr"].append(z_max)
                snippets["Analysis"]["d2 AccCorr"].append(d2z)
                snippets["Analysis"]["Pure Acc Corr"].append(za_max)
                snippets["Analysis"]["Arhy Corr"].append(arhy_cor)
                snippets["Analysis"]["Arhy Corr EKG"].append(arhy_cor_ekg)
                snippets["Analysis"]["Arhy Corr Quotient"].append(arhy_cor_quot)
                snippets["Analysis"]["Spectral Entropy ACC"].append(spec_ent_acc)

                (
                    acc_rms,
                    acc_kurt,
                    acc_skew,
                    acc_median,
                    acc_pp,
                    acc_pprms_ratio,
                    power_bands,
                    psd_mean,
                    psd_std,
                    psd_kurt,
                    psd_skew,
                    psd_max,
                    psd_maxfreq,
                ) = ac_f
                snippets["Analysis"]["Mean ACC Power"].append(acc_rms)
                snippets["Analysis"]["Mean ACC Kurtosis"].append(acc_kurt)
                snippets["Analysis"]["Mean ACC Skewness"].append(acc_skew)
                snippets["Analysis"]["Mean ACC Median"].append(acc_median)
                snippets["Analysis"]["Mean ACC PeaktoPeak"].append(acc_pp)
                snippets["Analysis"]["Mean ACC PP-RMS-Ratio"].append(acc_pprms_ratio)

                psd0, psd3, psd6, psd9, psd12, psd15 = power_bands
                snippets["Analysis"]["PSD_03_Band"].append(psd0)
                snippets["Analysis"]["PSD_36_Band"].append(psd3)
                snippets["Analysis"]["PSD_69_Band"].append(psd6)
                snippets["Analysis"]["PSD_912_Band"].append(psd9)
                snippets["Analysis"]["PSD_1215_Band"].append(psd12)
                snippets["Analysis"]["PSD_1518_Band"].append(psd15)

                snippets["Analysis"]["PSD_Mean"].append(psd_mean)
                snippets["Analysis"]["PSD_Std"].append(psd_std)
                snippets["Analysis"]["PSD_Kurt"].append(psd_kurt)
                snippets["Analysis"]["PSD_Skew"].append(psd_skew)
                snippets["Analysis"]["PSD_Max1"].append(psd_max)
                snippets["Analysis"]["PSD_Maxfreq1"].append(psd_maxfreq)

                [P_lea, Lmin, bS, nP, P_fib, P_h] = ekg_feat
                snippets["Analysis"]["Power LEA"].append(P_lea)
                snippets["Analysis"]["Min ECG Length"].append(Lmin)
                snippets["Analysis"]["bS"].append(bS)
                snippets["Analysis"]["number Peaks"].append(nP)
                snippets["Analysis"]["Fibrillation Power"].append(P_fib)
                snippets["Analysis"]["High Freq Power"].append(P_h)

                ekf_keys = [
                    "RR_Mean",
                    "RR_Std",
                    "QRS_PP_Mean",
                    "QRS_PP_Std",
                    "QRS_width_mean",
                    "QRS_width_std",
                    "QRS_heigth_width_ratio",
                    "ECG_norm_median",
                    "ECG_norm_var",
                    "ECG_slope_mean",
                    "ECG_slop_Std",
                    "ECG_slop_Kurtosis",
                    "AMSA",
                    "ECG_Fib_power",
                ]
                for key, value in zip(ekf_keys, ek_f):
                    snippets["Analysis"][key].append(value)
            i += 1

        k += 1
    return snippets


def av_mean(k: int, data: npt.ArrayLike) -> npt.NDArray:
    """
    Calculate a centered average mean of data over k elements.

    The data is zero-padded at the boundaries.
    """
    data = np.array(data)
    n = len(data)
    if k % 2 == 1:  # Calculate centered part of data
        erg = np.zeros(n - k + 1)
        for i in range(k):
            erg += data[i : n - k + i + 1]
        erg = erg / k
    else:
        erg = np.zeros(n - k)
        for i in range(k):
            erg += data[i : n - k + i]
        erg = erg / k
    erginit = np.zeros(k // 2)  # initial part
    ergfinal = np.zeros(k // 2)  # final part
    for i in range(k // 2):
        erginit[i] = (
            np.sum(data[: k // 2 + i + 1]) / k
        )  # since denominator ist still k, behaves like zero-padded data
        ergfinal[i] = np.sum(data[n - k // 2 - i - 1 :]) / (k)
    erg = np.append(erg, ergfinal)  # combine parts to one singal
    erg = np.insert(erg, 0, erginit)
    return erg


def max_nontriv_autocorr(
    acctime, acc
):  # Computes the nontrivial maximaum of the autocorrelation of signal
    sniplen = len(acc)
    z = np.correlate(acc, acc, mode="same") / (np.sum(np.square(acc)))
    a = z[2:] - z[1:-1]
    b = z[1:-1] - z[:-2]
    j = np.argwhere((b > 0) & (a < 0)).flatten() + 1
    j = j[np.abs(j - sniplen // 2) > 15]
    ze_max = 0
    j_max = 0
    for jj in j:
        if z[jj] > ze_max:
            ze_max = z[jj]
            j_max = jj

    return acctime[j_max] - 2, ze_max


def qrs_detector(ekg):
    ekg = np.array(ekg)
    but2 = sgn.butter(4, (0.5 / 125, 30 / 125), output="sos", btype="bandpass")
    ek_filt = sgn.sosfilt(but2, ekg)

    d_filt = av_mean(25, np.square(ek_filt[1:] - ek_filt[:-1]))  # 0.1 s average mean
    d_filt = np.append(d_filt, 0)
    d_filt = d_filt / np.max(d_filt)
    # Arhy Correlation via search for maxs
    d_filt2 = av_mean(
        25, np.square(ek_filt[1:] - ek_filt[:-1])
    )  # Search for QRS-komplexes ( idea after irusta above)
    d_filt2 = np.append(d_filt2, 0)
    d_filt2 = d_filt2 / np.max(d_filt2)

    max2 = (
        np.argwhere(
            (d_filt2[2:] - d_filt2[1:-1] < 0)
            & (d_filt2[1:-1] - d_filt2[:-2] > 0)
            & (d_filt2[1:-1] > 0.33)
        ).flatten()
        + 1
    )  # Search for maxima exceeding 0.33

    max3 = (
        np.argwhere(
            (ek_filt[2:] - ek_filt[1:-1]) * (ek_filt[1:-1] - ek_filt[:-2]) < 0
        ).flatten()
        + 1
    )
    # Compute true maxs by taking the largest local maximum within 50*0.004=0.200 seconds
    true_max = np.array([max3[0]])
    for elem in max2:
        cand = max3[np.abs(max3 - elem) < 75]
        if len(cand) == 0:
            cand = np.array([max3[np.argmin(np.abs(max3 - elem))]])
        ek_cand = np.abs(ek_filt[cand])

        true_cand = cand[np.argmax(ek_cand)]
        if (true_cand - true_max[-1]) < 75:
            if np.abs(ek_filt[true_cand]) > np.abs(ek_filt[true_max[-1]]):
                true_max[-1] = true_cand
        else:
            true_max = np.append(true_max, true_cand)

    n = len(ekg)
    le = 60
    true_max = true_max.astype(int)
    true_max = true_max[(true_max > le) & (true_max < n - le)]
    qq = np.array([], dtype=int)
    ss = np.array([], dtype=int)
    for r in true_max:
        ii = np.where(max3 == r)[0][0]
        q_cand = max3[ii - 4 : ii]
        q_cand = q_cand[r - q_cand < 100]
        if len(q_cand) == 0:
            q = 0
        else:
            if ek_filt[r] > 0:
                q = q_cand[np.argmin(ek_filt[q_cand])]
            if ek_filt[r] < 0:
                q = q_cand[np.argmax(ek_filt[q_cand])]
        s_cand = max3[ii + 1 : ii + 5]
        s_cand = s_cand[s_cand - r < 100]
        if len(s_cand) == 0:
            s = 0
        else:
            if ek_filt[r] > 0:
                s = s_cand[np.argmin(ek_filt[s_cand])]
            if ek_filt[r] < 0:
                s = s_cand[np.argmax(ek_filt[s_cand])]
        qq = np.append(qq, q)
        ss = np.append(ss, s)
    return true_max, qq, ss


def acc_feature(acc, acc_ensemble):  # Compute all ACC Features from Ashouri 2017
    if acc_ensemble.shape[0] == 120:
        accm = acc_ensemble
    else:
        accm = np.mean(acc_ensemble, axis=0)
    acc_rms = np.sqrt(np.sum(np.square(accm)) / 120)
    acc_kurt = stats.kurtosis(accm)
    acc_skew = stats.skew(accm)
    acc_median = np.median(accm)
    acc_pp = np.max(accm) - np.min(accm)
    acc_pprms_ratio = acc_pp / acc_rms

    freq, psd = sgn.welch(accm, fs=250, nperseg=120)
    psd = psd / np.sqrt(np.sum(np.square(psd)))
    cond = (freq > 0.8) & (freq < 30)
    psd = psd[cond]
    freq = freq[cond]

    power_bands = []
    for freq_thresh in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18)]:
        fl, fu = freq_thresh
        cond = (freq >= fl) & (freq < fu)
        psd_thres = psd[cond]
        power_bands.append(np.sqrt(np.sum(np.square(psd_thres))))

    freq, psd = sgn.welch(accm, fs=250, nperseg=120)
    cond = (freq > 0.8) & (freq < 30)
    psd = psd[cond]
    freq = freq[cond]
    psd_mean = np.mean(psd)
    psd_std = np.std(psd)
    psd_kurt = stats.kurtosis(psd)
    psd_skew = stats.skew(psd)
    psd_max = np.max(psd)
    psd_maxfreq = freq[np.argmax(psd)]
    acc_feature_list = [
        acc_rms,
        acc_kurt,
        acc_skew,
        acc_median,
        acc_pp,
        acc_pprms_ratio,
        power_bands,
        psd_mean,
        psd_std,
        psd_kurt,
        psd_skew,
        psd_max,
        psd_maxfreq,
    ]
    return acc_feature_list


def ek_feature(ekg):  # Compute a variety of ecg-features (mostly Elola 2020)
    n = len(ekg)
    but2 = sgn.butter(4, (0.5 / 125, 30 / 125), output="sos", btype="bandpass")
    ek_filt = sgn.sosfilt(but2, ekg)

    # Arhy Correlation via search for maxs
    d_filt2 = av_mean(
        25, np.square(ek_filt[1:] - ek_filt[:-1])
    )  # Search for QRS-komplexes ( idea after irusta above)
    d_filt2 = np.append(d_filt2, 0)
    d_filt2 = d_filt2 / np.max(d_filt2)

    true_max, qq, ss = qrs_detector(ekg)

    pp_amp = np.array([])
    qs_width = np.array([])

    for q, s in zip(list(qq), list(ss)):
        ek = ekg[q : s + 1]
        if len(ek) > 0:
            pp_amp = np.append(pp_amp, np.max(ek) - np.min(ek))
            qs_width = np.append(qs_width, (s - q) * 0.004)
        else:
            pp_amp = np.append(pp_amp, 0)
            qs_width = np.append(qs_width, 0)

    if len(true_max) > 1:
        pp_mean = np.mean(pp_amp)
        pp_std = np.std(pp_amp)
        qrs_width_mean = np.mean(qs_width)
        qrs_width_std = np.std(qs_width)
        qrs_pp_width_ratio = np.median(pp_amp / (qs_width + 1e-16))
    else:
        pp_mean = 0
        pp_std = 0
        qrs_width_mean = 0
        qrs_width_std = 0
        qrs_pp_width_ratio = 0

    if len(true_max) >= 2:
        rr_list = (true_max[1:] - true_max[:-1]) * 0.004
        rr_mean = np.mean(rr_list)
        rr_std = np.std(rr_list)
    else:
        rr_mean = 4
        rr_std = 0

    ek_filt2 = (ek_filt - np.min(ek_filt)) / (np.max(ek_filt) - np.min(ek_filt))
    ecg_norm_median = np.median(ek_filt2)
    ecg_norm_var = np.var(ek_filt2)

    d_filt = ek_filt[1:] - ek_filt[:-1]  # 0.1 s average mean
    d_filt = np.append(d_filt, 0)
    # d_filt=d_filt/np.max(d_filt)

    slope_mean = np.mean(np.abs(d_filt))
    slope_std = np.std(np.abs(d_filt))
    slope_kurt = stats.kurtosis(d_filt2)

    s_ecg = np.abs(
        fft.fft(ek_filt * sgn.windows.tukey(1000, alpha=0.0), norm="ortho")[: n // 2]
    )
    freq = fft.fftfreq(n, 0.004)[: n // 2]
    cond = (freq > 2) & (freq < 30)
    AMSA = np.sum(freq[cond] * s_ecg[cond])

    cond = (freq > 17.5) & (freq < 30)
    ecg_fib_pow = np.sum(np.square(s_ecg[cond]))  # *250/2000

    hjmob, hjcomp = antropy.hjorth_params(ek_filt)
    ecg_skew = stats.skew(ek_filt)
    ek_class2 = [
        rr_mean,
        rr_std,
        pp_mean,
        pp_std,
        qrs_width_mean,
        qrs_width_std,
        qrs_pp_width_ratio,
        ecg_norm_median,
        ecg_norm_var,
        slope_mean,
        slope_std,
        slope_kurt,
        AMSA,
        ecg_fib_pow,
        hjmob,
        hjcomp,
        ecg_skew,
    ]
    return ek_class2


def ekg_acc_corr1(acctime, acc, ekg, nperse=1000, window=""):  # Compute Features
    n = len(acc)
    w = np.ones(n)

    acc_amp = np.sqrt(np.sum(np.square(acc)))  # Mean of absolute values (v_1)
    ekg_amp = np.sqrt(np.sum(np.square(ekg)))  # Mean of absolute values (v_2)

    # Compute Spectral overlap of Frequencies between ftr2 and ftr
    sp_cor = []
    ftr = 20.01  ## Frequency Threshold
    ftr2 = 0.75

    Sdd = np.abs(fft.fft(acc * w, norm="ortho")[: n // 2])  # FFT of Acc
    See = np.abs(fft.fft(ekg * w, norm="ortho")[: n // 2])  # FFT of ECG
    freq0 = fft.fftfreq(nperse, 0.004)[: n // 2]  # Freqs for FFT

    Sdd = Sdd[
        (freq0 < ftr) & (freq0 > ftr2)
    ]  # Include only frequencies below threshold
    See = See[(freq0 < ftr) & (freq0 > ftr2)]

    Sdd = Sdd / np.sqrt(np.sum(np.square(np.abs(Sdd))))  # Normalize Spectra
    See = See / np.sqrt(np.sum(np.square(np.abs(See))))

    acc_v = np.abs(Sdd) - np.mean(np.abs(Sdd))  # Subtract mean from spectra
    ekg_v = np.abs(See) - np.mean(np.abs(See))

    acc_st2 = np.sum(np.square(acc_v))  # compute absolute of spectra
    ekg_st2 = np.sum(np.square(ekg_v))
    sp_cor = np.sum(acc_v * ekg_v) / np.sqrt(
        acc_st2 * ekg_st2
    )  # Compute spectral overlap ( Covariance of Spectra) v_3

    # Use Autocorrelation methods
    z = np.correlate(acc, acc, mode="same") / (
        np.sum(np.square(acc))
    )  # Acc Autocorrelation
    ze = np.correlate(ekg, ekg, mode="same") / (
        np.sum(np.square(ekg))
    )  # ECG Autocorrelation

    a = ze[2:] - ze[1:-1]  # First Difference forward
    b = ze[1:-1] - ze[:-2]  # First Difference backward
    j = np.argwhere((b > 0) & (a < 0)).flatten()  # Search maxima of ecg autocorr
    j = j[
        (np.abs(j - n // 2) > 75) & (j > 12) & (j < n - 12)
    ]  # Ignore maxima in the middle (trivial and at the boundaries)

    # Search for largest maximum of ECG autocorr
    ze_max = -1
    j_max = 0
    for jj in j:
        if ze[jj] > ze_max:
            ze_max = ze[jj]  # v_4
            j_max = jj
    if len(j) == 0:
        j_max = 250

    z_max = z[j_max]  # Acc Autocorr @ largest ECG autocorr v_6
    d2zz = (
        z[j_max + 1 - 2 : j_max + 1 + 3]
        + z[j_max - 1 - 2 : j_max - 1 + 3]
        - 2 * z[j_max - 2 : j_max + 3]
    ) * 250**2  # second derivative at this point
    d2z = np.nanmean(d2zz)  # v_7
    # d2z=(z[j_max+1]+z[j_max-1]-2*z[j_max])*250**2
    za_max = max_nontriv_autocorr(acctime, acc)[
        1
    ]  # Maximal autocorr of acceleration only v_5

    # Features nach Irusta 2014 "A Reliable Method for Rhythm Analysis during Cardiopulmonary Resuscitation"
    but = sgn.butter(5, 2.5 / 125, output="sos", btype="highpass")
    ek_lea = sgn.sosfilt(but, ekg)

    P_lea = np.sum(np.square(ek_lea))
    Lk = np.array([])
    for k in range(8):
        Lk = np.append(
            Lk, np.sum(np.sqrt(np.square(ek_lea[k * 125 : (k + 1) * 125]) + 0.004**2))
        )
    Lmin = np.min(Lk)

    but2 = sgn.butter(4, (0.5 / 125, 30 / 125), output="sos", btype="bandpass")
    ek_filt = sgn.sosfilt(but2, ekg)

    d_filt = av_mean(25, np.square(ek_filt[1:] - ek_filt[:-1]))  # 0.1 s average mean
    d_filt = np.append(d_filt, 0)
    d_filt = d_filt / np.max(d_filt)

    bS = np.percentile(d_filt, 10)
    maxs = (
        np.argwhere(
            (d_filt[2:] - d_filt[1:-1] < 0)
            & (d_filt[1:-1] - d_filt[:-2] > 0)
            & (d_filt[1:-1] > 0.2)
        ).flatten()
        + 1
    )
    nP = len(maxs)

    ffe0 = np.abs(fft.fft(ekg * sgn.windows.hamming(n), norm="ortho")[: n // 2])
    ffe0 = ffe0 / np.sqrt(np.sum(np.square(np.abs(ffe0))))

    P_fib = np.sum(np.square(ffe0[(freq0 >= 2.5) & (freq0 <= 7.5)]))
    P_h = np.sum(np.square(ffe0[(freq0 >= 12)]))

    ek_classifiers = [P_lea, Lmin, bS, nP, P_fib, P_h]

    # Arhy Correlation via search for maxs
    d_filt2 = av_mean(
        25, np.square(ek_filt[1:] - ek_filt[:-1])
    )  # Search for QRS-komplexes ( idea after irusta above)
    d_filt2 = np.append(d_filt2, 0)
    d_filt2 = d_filt2 / np.max(d_filt2)

    max2 = (
        np.argwhere(
            (d_filt2[2:] - d_filt2[1:-1] < 0)
            & (d_filt2[1:-1] - d_filt2[:-2] > 0)
            & (d_filt2[1:-1] > 0.33)
        ).flatten()
        + 1
    )  # Search for maxima exceeding 0.33

    # Compute true maxs by taking the largest local maximum within 50*0.004=0.200 seconds
    true_max = np.array([max2[0]])
    j = d_filt2[max2[0]]
    for elem in max2:
        if (elem - true_max[-1]) < 50:
            if d_filt2[elem] > d_filt2[true_max[-1]]:
                true_max[-1] = elem
        else:
            true_max = np.append(true_max, elem)

    le = 60
    true_max = true_max.astype(int)
    true_max = true_max[
        (true_max > le) & (true_max < n - le)
    ]  # Take only maxima, which allow cutting a window around them
    # not to close to the border

    i = 1
    # Compute Correlation between all windows
    corr_arhy = np.array([])
    corr_arhy_ekg = np.array([])

    for beat in true_max:
        for beat2 in true_max[i:]:
            ac1 = acc[beat - le : beat + le]
            ac2 = acc[beat2 - le : beat2 + le]
            corr_arhy = np.append(
                corr_arhy,
                np.sum(ac1 * ac2)
                / (np.sqrt(np.sum(np.square(ac1)) * np.sum(np.square(ac2)))),
            )

            ek1 = ekg[beat - le : beat + le]
            ek2 = ekg[beat2 - le : beat2 + le]
            corr_arhy_ekg = np.append(
                corr_arhy_ekg,
                np.sum(ek1 * ek2)
                / (np.sqrt(np.sum(np.square(ek1)) * np.sum(np.square(ek2)))),
            )

        i += 1
        # take 3rd quartile of this quantity
    if len(corr_arhy) > 0:
        arhy_cor = np.percentile(corr_arhy, 75)
    else:
        arhy_cor = 0
    if len(corr_arhy_ekg) > 0:
        arhy_cor_ekg = np.percentile(corr_arhy_ekg, 75)
        arhy_cor_quot = arhy_cor / arhy_cor_ekg
    else:
        arhy_cor_ekg = 0
        arhy_cor_quot = 0

    # make ensemble average

    if len(true_max) > 0:
        for i, beat in enumerate(true_max):
            ac1 = acc[beat - le : beat + le]
            if i == 0:
                acc_ensemble = np.array(ac1)
            else:
                acc_ensemble = np.vstack([acc_ensemble, ac1])
    else:
        acc_ensemble = np.random.randn(5, 120)
    spec_ent_acc = antropy.spectral_entropy(acc, sf=250, normalize=True, method="welch")

    ac_f = acc_feature(acc, acc_ensemble)
    ek_f = ek_feature(ekg)

    return (
        sp_cor,
        acc_amp,
        ekg_amp,
        z_max,
        ze_max,
        za_max,
        d2z,
        arhy_cor,
        arhy_cor_ekg,
        arhy_cor_quot,
        spec_ent_acc,
        ek_classifiers,
        ac_f,
        ek_f,
    )


def data_filter(
    X,
    X_background,
    y,
    accthresh=20,
    ecgthresh=2.5,
    accr=25,
    ecgr=35,
    arresttimemin=None,
    rosctimemin=None,
    wo_end=False,
    ros_co=None,
    are_co=None,
    shock_before=None,
    shock_after=None,
    background_keys=[
        "Start Time",
        "Start Distance",
        "End Distance",
        "Max ACC",
        "Max ECG",
        "Max CO2",
        "ACC ratio",
        "ECG ratio",
        "Shock before",
        "Shock after",
        "Rhythm",
    ],
):
    # Filters given features X, background infos X_background and labels y after conditions given in background infos
    cond = np.ones(X.shape[0], dtype=bool)
    # print(f"All_data: {len(cond[cond==1])}")
    i = background_keys.index("Max ACC")
    cond = (
        (cond) & (X_background.T[i] < accthresh)
    )  # max(abs(acc)) during a Snippet must be smaller than accthresh # Remove Noisy data
    # print(f"After ACC-thresh {len(cond[cond==1])}")

    i = background_keys.index("Max ECG")
    # print(f"ECG-Thresh-Values (Mean, Min Max) {np.mean(X_background.T[i])},{np.min(X_background.T[i])},{np.max(X_background.T[i])}")

    cond = (
        (cond) & (X_background.T[i] < ecgthresh)
    )  # max(abs(ecg)) during a Snippet must be smaller than ecgthresh # Remove Noisy data
    # print(f"After ECG-thresh {len(cond[cond==1])}")

    i = background_keys.index("ACC ratio")
    cond = (
        (cond) & (X_background.T[i] < accr)
    )  # max(abs(acc))/median(abs(acc)) during a Snippet must be smaller than accr. # Remove data where amplitudes are sometimes way bigger than otherwhere (sudden peaks etc)
    # print(f"After ACC-ratio {len(cond[cond==1])}")

    i = background_keys.index(
        "ECG ratio"
    )  # max(abs(ecg))/median(abs(ecg)) during a Snippet must be smaller than ecgr. # Remove data where amplitudes are sometimes way bigger than otherwhere (sudden peaks etc)
    cond = (cond) & (X_background.T[i] < ecgr)
    # print(f"After ECG-ratio{len(cond[cond==1])}")

    X = X[cond]
    X_background = X_background[cond]
    y = y[cond]

    return X, X_background, y


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
    asset_dir = Path(__file__).parent.parent / "assets"
    svc = joblib.load(asset_dir / "circ-classification_model.joblib")
    scaler = joblib.load(asset_dir / "circ-classification_scaler.joblib")
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


def determine_gaps_in_recording(
    time: npt.ArrayLike, data: npt.ArrayLike
) -> (npt.NDArray, npt.NDArray):
    dtime = time[1:] - time[:-1]
    median_dtime = np.nanmedian(dtime)
    gap_start_indices = np.nonzero(dtime > 2 * median_dtime)[0]
    gap_start = []
    gap_stop = []
    for start_index in gap_start_indices:
        starttime = time[start_index]
        stoptime = time[start_index + 1]
        gap_start.append(starttime)
        gap_stop.append(stoptime)
    return np.asarray(gap_start), np.asarray(gap_stop), np.array(gap_start_indices)


def linear_interpolate_gaps_in_recording(
    time: npt.ArrayLike, data: npt.ArrayLike
) -> (npt.NDArray, npt.NDArray):
    dtime = time[1:] - time[:-1]
    median_dtime = np.nanmedian(dtime)
    gap_start, gap_stop, gap_start_indices = determine_gaps_in_recording(time, data)
    for starttime, stoptime, start_index in zip(gap_start, gap_stop, gap_start_indices):
        time_in_gap = np.arange(starttime, stoptime, median_dtime)
        data_in_gap = data[start_index] + (
            data[start_index + 1] - data[start_index]
        ) / (stoptime - starttime) * (time_in_gap - starttime)
        time = np.insert(time, start_index, time_in_gap)
        data = np.insert(data, start_index, data_in_gap)
    return time, data
