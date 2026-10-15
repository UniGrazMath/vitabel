"""Reading defibrillator recordings: detection of the device / export format
from the given file (and the files next to it), and dispatch to the
corresponding loading routine."""

from __future__ import annotations

import logging
import os

from pathlib import Path

from vitabel.io._common import rename_channels
from vitabel.io.corpuls import read_corpuls
from vitabel.io.lifepak import read_lifepak
from vitabel.io.lucas import read_lucas
from vitabel.io.zoll import read_zollcsv, read_zolljson, read_zollxml
from vitabel.utils import constants

logger = logging.getLogger("vitabel")


def read_defibrillator_file(filepath: Path | str) -> tuple[dict, dict] | None:
    """Read a recording exported from a defibrillator (or mechanical CPR
    device), see :meth:`.Vitals.add_defibrillator_recording` for the
    supported devices and the files that are expected.

    Parameters
    ----------
    filepath
        The path to the exported file.

    Returns
    -------
    A tuple ``(pat_dat, dats)`` of the recording's (meta)data and a
    dictionary of the loaded data (pandas objects, keyed by channel name),
    or ``None`` if the file could not be read (unknown file type, or an
    empty LUCAS recording).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError("File not found in directory. Check path!")
    file_extension = filepath.suffix
    filename = filepath.stem
    dirpath = filepath.parent

    logger.info(f"Reading file {filename}")
    fileend_c = [file_extension]
    for file in dirpath.glob("*"):
        if filename in str(file):
            extension = file.suffix
            fileend_c.append(extension)

    logger.info(f"Endings {fileend_c}")

    if ".json" in fileend_c:
        pat_dat, dats = read_zolljson(filepath)
        dats = rename_channels(dats, constants.zoll2channelnames_dict)
        logger.info(f"File: {filename} successfully read!")
    elif (".xml" in fileend_c) and (".txt" not in fileend_c):
        if "_Continuous" in filename:
            pure_filename = filename[: filename.index("_Continuous")]
            fp1 = dirpath.joinpath(Path(pure_filename + "_Continuous.xml"))
            fp2 = dirpath.joinpath(Path(pure_filename + "_Continuous_Waveform.xml"))
            fp3 = dirpath.joinpath(Path(pure_filename + "_CprEventLog.xml"))

            further_files = []
            for file in dirpath.glob("*.xml"):
                file = str(file)
                if (
                    pure_filename in file
                    and "_Continuous.xml" not in file
                    and "_Continuous_Waveform.xml" not in file
                    and "_CprEventLog.xml" not in file
                    and ".xml" in file
                ):
                    file_ending = file[
                        file.find(filename) + len(filename) : file.rfind(".")
                    ]
                    further_files.append(file_ending)

            if further_files:
                logger.warning(
                    f"Warning! Further Lifepak-Files {further_files} for recording {pure_filename} found in directory. These files are not loaded to the cardio-Object currently. Please modify the loading routine to include them"
                )
            if not (
                os.path.isfile(fp1) and os.path.isfile(fp2) and os.path.isfile(fp3)
            ):
                raise FileNotFoundError(
                    f"Error when Loading LIFEPAK Recording! Expected Files {fp2} and {fp3}, which are not found, and the recording cannot be loaded. Check Export of your LIFEPAK_Recording"
                )

            else:
                pat_dat, dats = read_lifepak(fp1, fp2, fp3, further_files=further_files)
                dats = rename_channels(dats, constants.LP2channelnames_dict)
            logger.info(f"LIFEPAK-File: {str(filename)} successfully read!")
        elif "_Lucas" in filename:
            pure_filename = filename[: filename.index("_Lucas")]

            fp1 = dirpath.joinpath(Path(pure_filename + "_Lucas.xml"))
            fp2 = dirpath.joinpath(Path(pure_filename + "_CprEventLog.xml"))
            if not (os.path.isfile(fp1) and os.path.isfile(fp2)):
                raise FileNotFoundError(
                    f"Error when Loading LUCAS Recording! Expected Files additional file {fp2}, which is not found, and the recording cannot be loaded. Check Export of your LUCAS-Recording"
                )

            else:
                pat_dat, dats = read_lucas(fp1, fp2)
                dats = rename_channels(dats, constants.LP2channelnames_dict)

            if dats:
                logger.info(f"LUCAS-File: {str(filename)} successfully read!")
            else:
                logger.warning(f"LUCAS-File: {str(filename)} is empty!")
                return None

        else:
            pat_dat, dats = read_zollxml(dirpath.joinpath(Path(filename + ".xml")))
            dats = rename_channels(dats, constants.zoll2channelnames_dict)

            logger.info(f"File: {filename}.xmlsuccessfully read!")
    elif ".txt" in fileend_c:
        pure_filename = filename[: filename.index("_ecg")]

        pat_dat, dats = read_zollcsv(
            dirpath.joinpath(Path(pure_filename + "_ecg.txt")),
            dirpath.joinpath(Path(pure_filename + ".xml")),
        )
        dats = rename_channels(dats, constants.zoll2channelnames_dict)

        logger.info(f"File: {filename} successfully read!")
    elif ".bdf" in fileend_c:
        pat_dat, dats = read_corpuls(dirpath.joinpath(Path(filename + ".bdf")))
        dats = rename_channels(dats, constants.corpuls2channelnames_dict)

        logger.info(f"File: {filename} successfully read!")

    elif fileend_c != []:
        logger.error(f"Error: No method to read {fileend_c} files!")
        return None

    return pat_dat, dats
