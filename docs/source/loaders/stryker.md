# Stryker / Physio-Control (CodeStat)

Recordings of Stryker devices are exported to XML with the *CodeStat* software
(versions 11 and 12) and loaded with `Vitals.add_defibrillator_recording`. All
files of a recording share a prefix and have to be in the same directory; pass
the path of the entry file.

| Device | Entry file | Further files |
|---|---|---|
| LIFEPAK 15 | `<prefix>_Continuous.xml` | `<prefix>_Continuous_Waveform.xml`, `<prefix>_CprEventLog.xml` (required); `<prefix>_TrendData.xml` (optional) |
| CPR feedback device (e.g., TrueCPR) | `<prefix>_CprEventLog.xml` | – |
| LUCAS | `<prefix>_Lucas.xml` | `<prefix>_CprEventLog.xml` |

The XML namespace used by CodeStat 12 is handled transparently.

## LIFEPAK 15

### Waveforms

From `_Continuous_Waveform.xml`. Only the channels shown on the device are
recorded, so a waveform may have gaps (e.g., when the displayed ECG lead is
switched). Values are multiplied by the scale factor given in the file.

| CodeStat channel | vitabel channel |
|---|---|
| `Paddles`, `Paddles (Generic)` | `ecg_pads` |
| `I`, `II` | `ecg_i`, `ecg_ii` |
| `Impedance` | `impedance` |
| `CO2` | `capnography` |
| `SpO2` | `ppg` |
| `IP1`, `IP2` | `ibp_1`, `ibp_2` |

Other channels (e.g., further ECG leads) keep their CodeStat names.

### Trend values

Vital signs measured by the device at irregular times, from the events of
`_CprEventLog.xml`, or from `_TrendData.xml` if it is present (it contains
several times more values). For values with a status (invasive blood
pressure), only the valid ones are kept.

| CodeStat value | vitabel channel | Unit |
|---|---|---|
| `VitalsHeartRate` | `heart_rate` | 1/min |
| `VitalsCO2` | `etco2` | mmHg |
| `VitalsRespRate` | `respiratory_rate` | 1/min |
| `VitalsFICO2` | `mean_inspired_co2` | % |
| `VitalsSPO2Saturation` | `spo2` | % |
| `VitalsSPO2PulseRate` | `heart_rate_ppg` | 1/min |
| `VitalsNIBPSystolic`, `...Diastolic`, `...Mean` | `nibp_sys`, `nibp_dia`, `nibp_map` | mmHg |
| `VitalsNIBPPulseRate` | `heart_rate_nibp` | 1/min |
| `VitalsIP1Systolic`, `...Diastolic`, `...Mean` | `ibp_1_sys`, `ibp_1_dia`, `ibp_1_map` | mmHg |
| `VitalsPaddlesImpedance` | `impedance_2` | Ohm |
| `VitalsAmbientPressure` | `ambient_pressure` | mmHg |

Further values (e.g., `VitalsSpCO`, `VitalsIP2Systolic`) keep their CodeStat
names.

### CPR events and defibrillations

From `_CprEventLog.xml`; channels with time points only, except for the
defibrillations.

| CodeStat event | vitabel channel |
|---|---|
| `ChestCompression` | `cc` |
| `Ventilation` | `ventilations` |
| `StartCPR`, `StopCPR` | `start_cpr`, `stop_cpr` |
| `12Lead` | `time_12_lead_ecg` |
| `Defib` | `defibrillations_DeliveredEnergy` (J), `defibrillations_Impedance` (Ohm) |

A shock without an energy or impedance value gets `NaN` for it.
`Vitals.shocks()` collects the defibrillations in one table. If the number of
shocks in the event log differs from the total reported by the device, the
defibrillations are not imported (and an error is logged).

## CPR feedback devices (`_CprEventLog.xml` only)

Exports of CPR feedback devices contain the `DeviceCompression` events only;
compressions deleted by the user in CodeStat are skipped.

| vitabel channel | Content | Unit |
|---|---|---|
| `cc` | time of each compression | – |
| `cc_depth` | compression depth | cm |
| `cc_rate` | compression rate | 1/min |

The units are the same as for the compression data of ZOLL recordings, so the
CPR analyses (e.g., `Vitals.cycle_duration_analysis`, `make_cprdat_analysis`)
work in the same way.

## LUCAS

From `_Lucas.xml` (device information, shocks) and `_CprEventLog.xml`: the
compressions (`cc`) and defibrillations, as for LIFEPAK 15.
