# Device loaders

`vitabel` reads the exports of the following devices directly. Further data
can be added from CSV files, pandas DataFrames, or dictionaries
(`Vitals.add_data_from_csv`, `Vitals.add_data_from_DataFrame`,
`Vitals.add_data_from_dict`).

## Supported devices

| Device | Export | File(s) of the export | Pass to | Method |
|---|---|---|---|---|
| ZOLL X Series, R Series | from the device, as JSON or XML | `<name>.json` or `<name>.xml` | that file (also compressed, `.json.bz2` / `.xml.bz2`) | `add_defibrillator_recording` |
| ZOLL E Series, AED Pro | from the device | `<name>_ecg.txt`, `<name>.xml` | `<name>_ecg.txt` | `add_defibrillator_recording` |
| Stryker LIFEPAK 15 | CodeStat 11 or 12 (XML) | `<prefix>_Continuous.xml`, `<prefix>_Continuous_Waveform.xml`, `<prefix>_CprEventLog.xml`; optionally `<prefix>_TrendData.xml` | `<prefix>_Continuous.xml` | `add_defibrillator_recording` |
| Stryker CPR feedback devices (e.g. TrueCPR) | CodeStat (XML) | `<prefix>_CprEventLog.xml` | that file | `add_defibrillator_recording` |
| Stryker LUCAS | CodeStat (XML) | `<prefix>_Lucas.xml`, `<prefix>_CprEventLog.xml` | `<prefix>_Lucas.xml` | `add_defibrillator_recording` |
| Corpuls | Corpuls analysis software (BDF) | `<name>.bdf`, and a directory `<name>` with the event logs next to it | `<name>.bdf` | `add_defibrillator_recording` |
| EOlife (ventilation feedback) | CSV export | `<name>.csv` | that file | `add_ventilatory_feedback` |
| VitalDB (Vital Recorder) | `.vit` recording | `<name>.vit` | that file | `add_vital_db_recording` |
| EDF+ recordings | EDF+ file | `<name>.edf` | that file | `add_edfplus` |

All files of an export have to be in the same directory. Which device a
defibrillator export comes from is detected from the file names (and, for
ZOLL, from the serial number of the device).

The tests of `vitabel` use exports of the ZOLL X Series (JSON), LIFEPAK 15
(CodeStat 11; CodeStat 12 and CPR feedback exports are derived from it),
LUCAS, EOlife, VitalDB, and EDF+ files. There is no test data for ZOLL XML,
ZOLL E Series / AED Pro (only a synthetic case), R Series, and Corpuls exports;
please report problems with these formats.

## Details

```{toctree}
:maxdepth: 1
loaders/stryker
```
