# Stryker / Physio Control LIFEPAK — Loader Reference

Covers data exported from **LIFEPAK 12**, **LIFEPAK 15**, and **LP1000** devices via
Stryker's *CodeStat* software, as well as standalone exports from the **TrueCPR**
feedback device. Data are exported as XML and loaded via
`Vitals.add_defibrillator_recording`.

---

## Export formats and entry points

| Export type | Entry-point file | Additional required files |
|---|---|---|
| Full defibrillator recording | `<prefix>_Continuous.xml` | `<prefix>_Continuous_Waveform.xml`, `<prefix>_CprEventLog.xml` |
| CPR-device-only (TrueCPR) | `<prefix>_CprEventLog.xml` | — |
| LUCAS mechanical CPR device | `<prefix>_Lucas.xml` | `<prefix>_CprEventLog.xml` |

Pass the **entry-point file** to `add_defibrillator_recording`. All other required
files are discovered automatically in the same directory.

> **Warning (CPR-device-only):** When loading a standalone `_CprEventLog.xml`, a
> logger warning is emitted and no waveform data is available.

---

## XML structure

```
<prefix>_Continuous.xml        — device metadata, trend vitals, events (CPR, shocks)
<prefix>_Continuous_Waveform.xml — waveform data per lead/channel
<prefix>_CprEventLog.xml       — per-compression events with depth and rate
```

All files share the XML namespace `http://www.physiocontrol.com/code-stat/research-exporter`.
The parser extracts this namespace from the root element tag at runtime
(`pcstr = "{http://...}"`) and prefixes all `ElementTree` lookups accordingly.

---

## Waveform channels

Sourced from `_Continuous_Waveform.xml` → `Record/RecordData` elements.
Each `RecordData` element carries one channel with one or more `Waveforms` (segments).
Segments may be non-contiguous (LIFEPAK only records displayed channels; switching
the displayed lead creates a recording gap).

These become `Channel` objects in vitabel.

| XML `Channel` name | vitabel name | Unit | Sample rate | Notes |
|---|---|---|---|---|
| `Paddles` | `Paddles` | mV | 125 Hz | Defibrillator pads (generic paddle type) |
| `Combo Pads` | `Combo Pads` | mV | 125 Hz | QuickCombo adhesive pads |
| `I` | `ecg_i` | mV | 125 Hz | ECG lead I |
| `II` | `ecg_ii` | mV | 125 Hz | ECG lead II |
| `III` | `III` | mV | 125 Hz | ECG lead III (rename not yet in `LP2channelnames_dict`) |
| `aVR` | `aVR` | mV | 125 Hz | Augmented lead (rename pending) |
| `aVL` | `aVL` | mV | 125 Hz | Augmented lead (rename pending) |
| `aVF` | `aVF` | mV | 125 Hz | Augmented lead (rename pending) |
| `V1` | `V1` | mV | 125 Hz | Precordial lead (rename pending) |
| `V3` | `V3` | mV | 125 Hz | Precordial lead (rename pending) |
| `V5` | `V5` | mV | 125 Hz | Precordial lead (rename pending) |
| `V6` | `V6` | mV | 125 Hz | Precordial lead (rename pending) |
| `Impedance` | `impedance` | Ohm | 60–61 Hz | Thoracic impedance (transthoracic) |
| `CO2` | `capnography` | mmHg | 40 Hz | CO₂ waveform (capnogram) |
| `SpO2` | `ppg` | — | — | SpO₂ photoplethysmography waveform (rare in exports) |

> **Note:** `LP2channelnames_dict` currently maps `Paddles (Generic)` → `ecg_pads`
> but CodeStat research exports use the shorter name `Paddles`, so the rename does
> not fire. This is a **known gap** — see open issues. Plot styles (`DEFAULT_PLOT_STYLE`)
> for `Paddles` and `Combo Pads` are also missing and need to be added.

---

## Trend / discrete-measurement channels

Sourced from `_Continuous.xml` → `Events/Event` elements whose `Values` carry
`VitalsXxx` sub-elements. Sampled at irregular intervals (device measurement cadence,
not a fixed rate).

These are currently loaded as `Channel` objects but should be **`Label` objects**
(numeric, global) — they are device-sampled discrete values, not continuous waveforms,
and are conceptually analogous to the MAP label in use case 3 of the vitabel paper.

| XML `Value[@Type]` | vitabel name | Unit | Notes |
|---|---|---|---|
| `VitalsHeartRate` | `heart_rate` | 1/min | Derived from ECG |
| `VitalsCO2` | `etco2` | mmHg | End-tidal CO₂ (per breath) |
| `VitalsRespRate` | `respiratory_rate` | 1/min | Respiratory rate |
| `VitalsFICO2` | `mean_inspired_co2` | mmHg | Fraction inspired CO₂ |
| `VitalsAmbientPressure` | `ambient_pressure` | mmHg | Barometric pressure |
| `VitalsSPO2Saturation` | `spo2` | % | Pulse oximetry saturation |
| `VitalsSPO2PulseRate` | `heart_rate_ppg` | 1/min | Pulse rate from SpO₂ probe |
| `VitalsSpCO` | `VitalsSpCO` | % | Carboxyhemoglobin (Masimo rainbow; rename pending) |
| `VitalsNIBPSystolic` | `nibp_sys` | mmHg | Non-invasive BP systolic |
| `VitalsNIBPDiastolic` | `nibp_dia` | mmHg | Non-invasive BP diastolic |
| `VitalsNIBPMean` | `nibp_map` | mmHg | Non-invasive BP mean |
| `VitalsNIBPPulseRate` | `heart_rate_nibp` | 1/min | Pulse rate from NIBP cuff |
| `VitalsPaddlesImpedance` | `impedance_2` | Ohm | Impedance measured at shock delivery |
| `VitalsIP1Systolic` | `ibp_1_sys` | mmHg | Invasive BP channel 1 systolic |
| `VitalsIP1Diastolic` | `ibp_1_dia` | mmHg | Invasive BP channel 1 diastolic |
| `VitalsIP1Mean` | `ibp_1_map` | mmHg | Invasive BP channel 1 mean |
| `CompressionsPerCPRInterval` | `compressions_per_cpr_interval` | — | Per-interval CPR metric |
| `VentillationsPerCPRInterval` | `ventillations_per_cpr_interval` | — | Per-interval CPR metric |
| `VentPauseDuration` | `vent_pause_duration` | s | Ventilation pause duration |
| `Age` | `age` | years | Patient age stored on device |
| `Airway` | `airway` | — | Airway device type (string) |

---

## Event channels

Sourced from `_CprEventLog.xml` → `Events/Event` elements.

These are currently loaded as `Channel` objects but should be refactored:
- Point events → **`Label`** (global, time-only or numeric)
- CPR periods → **`IntervalLabel`** (pairing `start_cpr` + `stop_cpr`)

| Event `Type` | vitabel name | vitabel type (target) | Notes |
|---|---|---|---|
| `ChestCompression` | `cc` | `Label` (time-only) | One entry per compression |
| `Ventilation` | `ventilations` | `Label` (time-only) | One entry per ventilation |
| `StartCPR` + `StopCPR` | `start_cpr` / `stop_cpr` → `cpr_periods` | `IntervalLabel` | Should be merged into one interval label |
| `Defib` → `DefibEnergy` | `defibrillations_DeliveredEnergy` | `Label` (numeric) | Joules; currently split across two channels |
| `Defib` → `DefibVoltageCompImpedance` | `defibrillations_Impedance` | `Label` (numeric) | Ohm; `NaN` when not recorded (e.g. older devices) |
| `12Lead` | `time_12_lead_ecg` | `Label` (time-only) | Timestamp of 12-lead ECG acquisition |

---

## TrueCPR standalone export (`_CprEventLog.xml` only)

When `_CprEventLog.xml` is the only available file (no `_Continuous.xml`), the
`read_lifepak_cpreventlog` function is used. It extracts `DeviceCompression` events.
Compressions flagged `DeletedByUser=Yes` are skipped.

| Source | vitabel name | vitabel type (target) | Unit | Notes |
|---|---|---|---|---|
| `DeviceCompression` timestamp | `cc` | `Label` (time-only) | — | |
| `DeviceCompression/Values/Depth` | `CompressionDepth` | `Label` (numeric) | mm | Per-compression depth |
| `DeviceCompression/Values/Rate` | `CompressionRate` | `Label` (numeric) | 1/min | Instantaneous rate at compression |

Device metadata (`Model`, `SerialNumber`, start time) is read from the root-level
`RecordingDevice` and `AdjustedPowerOn` elements.

---

## Known gaps and open issues

1. **Rename dict incomplete** — `LP2channelnames_dict` is missing entries for `Paddles`,
   `Combo Pads`, `III`, `aVF`, `aVL`, `aVR`, `V1`–`V6`, `VitalsSpCO`. These channels
   retain their XML names instead of receiving standardised vitabel names.

2. **Plot styles missing** — `DEFAULT_PLOT_STYLE` has no entries for `Paddles`,
   `Combo Pads`, `impedance_2`, `spo2`, `heart_rate`, `nibp_*`, and the new
   `CompressionDepth` / `CompressionRate` channels.

3. **Channels vs Labels** — All defibrillator-recorded events and discrete measurements
   are currently loaded as `Channel` objects. Per the vitabel data model, discrete
   events (`cc`, `ventilations`, shocks, 12-lead timestamps) should be `Label` objects,
   CPR periods should be an `IntervalLabel`, and trend vitals should be numeric `Label`
   objects. Refactoring `add_defibrillator_recording` and all loader return types is
   tracked as a separate issue.

4. **Impedance sample rate** — The `Impedance` channel sample rate varies slightly
   across devices (60 Hz on LP1000, ~61.038 Hz on LP15). The XML value is used
   directly.

---

## Loader implementation

| Function | File | Purpose |
|---|---|---|
| `read_lifepak(f_cont, f_cont_wv, f_cpre)` | `utils/loading.py` | Full LIFEPAK recording |
| `read_lifepak_cpreventlog(f_cpre)` | `utils/loading.py` | TrueCPR standalone |
| `read_lucas(f_luc, f_cpre)` | `utils/loading.py` | LUCAS mechanical CPR device |

Dispatch happens in `Vitals.add_defibrillator_recording` (`vitals.py`) based on the
filename stem:

```
_Continuous  → read_lifepak
_Lucas       → read_lucas
_CprEventLog → read_lifepak_cpreventlog
```
