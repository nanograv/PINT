"""Tools for working with pulse time-of-arrival (TOA) data.

In particular, single TOAs are represented by :class:`pint.toa.TOA` objects, and if you
want to manage a collection of these we recommend you use a :class:`pint.toa.TOAs` object
as this makes certain operations much more convenient. You probably want to load one with
:func:`pint.toa.get_TOAs` (from a ``.tim`` file) or :func:`pint.toa.get_TOAs_array` (from a
:class:`numpy.ndarray` or :class:`astropy.time.Time` object).

Warning
-------
Function:

- :func:`pint.simulation.make_fake_toas`

has moved to :mod:`pint.simulation`.
"""

import contextlib
import copy
import gzip
import pickle
import re
import warnings
from collections.abc import MutableMapping
from pathlib import Path

import astropy.table as table
import astropy.time as time
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    ICRS,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
)
from loguru import logger as log

import pint
import pint.utils
from pint.observatory import Observatory, bipm_default, get_observatory
from pint.observatory.special_locations import T2SpacecraftObs
from pint.observatory.satellite_obs import SatelliteObs
from pint.observatory.topo_obs import TopoObs
from pint.phase import Phase
from pint.pulsar_ecliptic import PulsarEcliptic
from pint.pulsar_mjd import Time
from pint.solar_system_ephemerides import objPosVel_wrt_SSB

__all__ = [
    "TOAs",
    "get_TOAs",
    "get_TOAs_list",
    "get_TOAs_array",
    "load_pickle",
    "save_pickle",
    "format_toa_line",
    "TOA",
]

EPHEM_default = "DE421"

toa_commands = (
    "DITHER",
    "EFAC",
    "EMAX",
    "EMAP",
    "EMIN",
    "EQUAD",
    "FMAX",
    "FMIN",
    "INCLUDE",
    "INFO",
    "JUMP",
    "MODE",
    "NOSKIP",
    "PHA1",
    "PHA2",
    "PHASE",
    "SEARCH",
    "SIGMA",
    "SIM",
    "SKIP",
    "TIME",
    "TRACK",
    "ZAWGT",
    "FORMAT",
    "END",
)

all_planets = ("jupiter", "saturn", "venus", "uranus", "neptune", "earth")

tempo_aliases = {
    "arecibo": "AO",
    "jodrell": "JB",
    "A1DOT": "XDOT",
    "STIGMA": "VARSIGMA",
    "EFAC": "T2EFAC",
    "EQUAD": "T2EQUAD",
    "ECORR": "TNECORR",
}


def get_TOAs(
    timfile,
    ephem=None,
    include_bipm=None,
    bipm_version=None,
    include_gps=None,
    planets=None,
    include_pn=True,
    model=None,
    usepickle=False,
    tdb_method="default",
    picklefilename=None,
    limits="warn",
):
    """Load and prepare TOAs for PINT use.

    This is the primary function for loading TOAs from a file.

    Loads TOAs from a ``.tim`` file, applies clock corrections, computes
    key values (like TDB), computes the observatory position and velocity
    vectors, and pickles the file for later use (if requested).

    Although ``.tim`` files are intended to be quite specific measurements,
    selecting one of the optional arguments to this function while loading the
    ``.tim`` file changes the interpretation; the ``.tim`` file represents different
    times if different settings are chosen. For nanosecond-level
    reproducibility it is necessary to specify the options with which each ``.tim``
    file was loaded. Observatory clock corrections are also applied, and thus
    the exact result also depends on the precise values in observatory clock
    correction files; normally these do not change.

    Note also that if usepickle is set, the pickled file will have clock
    corrections and other values set from when it was loaded and these may not
    correspond to the values you set here.

    See :func:`pint.toa.TOAs.apply_clock_corrections` for further information on the meaning of
    the clock correction flags.

    If commands like ``TIME`` or ``EQUAD`` are present in the ``.tim`` file,
    they are applied to the TOAs upon reading and retained in the ``.commands``
    attribute. Modern usage is to place ``EQUAD`` and ``EFAC`` in the ``.par``
    file instead, where they can be fit; these are supported here for
    historical reasons.

    Parameters
    ----------
    timfile : str or list of strings or file-like
        Filename, list of filenames, or file-like object containing the TOA data.
    ephem : str or None
        The name of the solar system ephemeris to use; defaults to the EPHEM parameter
        in the timing model (`model`) if it is given, otherwise defaults to ``pint.toa.EPHEM_default``.
    include_bipm : bool or None
        Whether to apply the BIPM clock correction. Defaults to True.
    bipm_version : str or None
        Which version of the BIPM tables to use for the clock correction.
        The format must be 'BIPMXXXX' where XXXX is a year.
    include_gps : bool or None
        Whether to include the GPS clock correction. Defaults to True.
    planets : bool or None
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.
    include_pn : bool, optional
        Whether or not to read in the 'pn' column (``pulse_number``)
    model : pint.models.timing_model.TimingModel or None
        If a valid timing model is passed, model commands (such as BIPM version,
        planet shapiro delay, and solar system ephemeris) that affect TOA loading
        are applied. The solar system ephemeris is superseded by the `ephem` parameter.
    usepickle : bool
        Whether to try to use pickle-based caching of loaded clock-corrected TOAs objects.
    tdb_method : str
        Which method to use for the clock correction to TDB. See
        :func:`pint.observatory.Observatory.get_TDBs` for details.
    picklefilename : str or None
        Filename to use for caching loaded file. Defaults to adding ``.pickle.gz`` to the
        filename of the timfile, if there is one and only one. If no filename is available,
        or multiple filenames are provided, a specific filename must be provided.
    limits : "warn" or "error"
        What to do when encountering TOAs for which clock corrections are not available.

    Returns
    -------
    TOAs
        Completed TOAs object representing the data.
    """
    if model:
        # If the keyword args are set, override what is in the model
        if ephem is None and model["EPHEM"].value is not None:
            ephem = model["EPHEM"].value
            log.debug(f"Using EPHEM = {ephem} from the given model")
        if include_bipm is None and model["CLOCK"].value is not None:
            if model["CLOCK"].value == "TT(TAI)":
                include_bipm = False
                log.info("Using CLOCK = TT(TAI), so setting include_bipm = False")
            elif "BIPM" in model["CLOCK"].value:
                clk = model["CLOCK"].value.strip(")").split("(")
                if len(clk) == 2:
                    ctype, cvers = clk
                    if ctype == "TT" and cvers.startswith("BIPM"):
                        include_bipm = True
                        if bipm_version is None:
                            if cvers == "BIPM":
                                bipm_version = bipm_default
                            else:
                                bipm_version = cvers
                                log.debug(
                                    f"Using CLOCK = {bipm_version} from the given model"
                                )
                    else:
                        log.warning(
                            f'CLOCK = {model["CLOCK"].value} is not implemented. '
                            f"Using TT({bipm_default}) instead."
                        )
            else:
                log.warning(
                    f'CLOCK = {model["CLOCK"].value} is not implemented. '
                    f"Using TT({bipm_default}) instead."
                )
        if (
            planets is None
            and "PLANET_SHAPIRO" in model
            and model["PLANET_SHAPIRO"].value
        ):
            planets = True
            log.debug("Using PLANET_SHAPIRO = True from the given model")

    updatepickle = False
    recalc = False
    if usepickle:
        try:
            t = load_pickle(timfile, picklefilename=picklefilename)
            log.info(f"Reading TOAs from the picklefile for `{timfile}`")
        except IOError:
            # Pickle either did not exist or is out of date
            updatepickle = True
        else:
            if hasattr(t, "hashes"):
                try:
                    if not t.check_hashes():
                        updatepickle = True
                        log.warning("Pickle file is based on files that have changed")
                except FileNotFoundError:
                    updatepickle = True
                    log.warning(
                        "FileNotFoundError when trying to check pickle hashes. Recomputing pickle."
                    )
            else:
                # Only pre-v0.8 pickles lack hashes.
                updatepickle = True
                log.info("Pickle is very old")
            if (
                include_gps is not None
                and t.clock_corr_info.get("include_gps", None) != include_gps
            ):
                log.info("Pickle contains wrong include_gps")
                updatepickle = True
            if (
                include_bipm is not None
                and t.clock_corr_info.get("include_bipm", None) != include_bipm
            ):
                log.info("Pickle contains wrong include_bipm")
                updatepickle = True
            if (
                bipm_version is not None
                and t.clock_corr_info.get("bipm_version", None) != bipm_version
            ):
                log.info("Pickle contains wrong bipm_version")
                updatepickle = True
    if not usepickle or updatepickle:
        if isinstance(timfile, (str, Path)) or hasattr(timfile, "readlines"):
            t = TOAs(timfile)
        else:
            t = merge_TOAs([TOAs(t) for t in timfile])

        files = [t.filename] if isinstance(t.filename, (str, Path)) else t.filename
        if files is not None:
            t.hashes = {f: pint.utils.compute_hash(f) for f in files}
        recalc = True

    if all("clkcorr" not in f for f in t.table["flags"]):
        if include_gps is None:
            include_gps = True
        if bipm_version is None:
            bipm_version = bipm_default
        if include_bipm is None:
            include_bipm = True
        # FIXME: should we permit existing clkcorr flags?
        t.apply_clock_corrections(
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            limits=limits,
        )

    if ephem is None:
        ephem = t.ephem
    elif ephem != t.ephem:
        if t.ephem is not None:
            # If you read a .tim file using TOAs(), the ephem is None
            # and so no recalculation is needed, just calculation!
            log.info("Ephem changed, recalculation needed")
        recalc = True
        updatepickle = True
    if recalc or "tdb" not in t.table.colnames:
        t.compute_TDBs(method=tdb_method, ephem=ephem)
    if planets is None:
        planets = t.planets
    elif planets != t.planets:
        log.debug("Planet PosVels will be calculated.")
        recalc = True
        updatepickle = True
    if recalc or "ssb_obs_pos" not in t.table.colnames:
        t.compute_posvels(ephem, planets)

    if usepickle and updatepickle:
        log.info("Pickling TOAs.")
        save_pickle(t, picklefilename=picklefilename)
    if "pulse_number" in t.table.colnames and not include_pn:
        log.warning("'pulse_number' column exists but not being read in")
        t.remove_pulse_numbers()

    dm_data, valid_data = t.get_flag_value("pp_dm", as_type=float)
    if len(valid_data) not in [0, len(t)]:
        raise ValueError(
            "Mixing narrowband and wideband toas in a TOAs object is not allowed. "
            "Make sure that either all or no TOAs have the -pp_dm flag."
        )

    return t


def load_pickle(toafilename, picklefilename=None):
    """Load a pickle file, un-gzipping if necessary.

    Parameters
    ----------
    toafilename : str
        Base filename of the TOAs; pickles will be searched for with
        ".pickle.gz", ".pickle", or just this filename.
    picklefilename : str, optional
        Explicit filename to use.

    Returns
    -------
    toas : :class:`pint.toa.TOAs`

    Raises
    ------
    IOError
        If no pickle is found.
    """
    picklefilenames = (
        [toafilename + ext for ext in (".pickle.gz", ".pickle", "")]
        if picklefilename is None
        else [picklefilename]
    )

    lf = None
    for fn in picklefilenames:
        with contextlib.suppress(IOError, pickle.UnpicklingError, ValueError):
            with gzip.open(fn, "rb") as f:
                lf = pickle.load(f)
        with contextlib.suppress(IOError, pickle.UnpicklingError, ValueError):
            with open(fn, "rb") as f:
                lf = pickle.load(f)
    if lf is not None:
        lf.was_pickled = True
        return lf
    raise IOError("No readable pickle found")


def save_pickle(toas, picklefilename=None):
    """Write the TOAs to a ``.pickle.gz`` file.

    Parameters
    ----------
    toas : :class:`pint.toa.TOAs`
        The TOAs to pickle.
    picklefilename : str, optional
        The filename to use for the pickle file; if not specified,
        construct a filename based on the file the toas object was
        originally loaded from.
    """
    # Save the PINT version used to create this pickle file
    toas.pintversion = pint.__version__
    if picklefilename is not None:
        pass
    elif toas.merged:
        raise ValueError(
            "TOAs object was merged from multiple files, please provide a filename."
        )
    elif toas.filename is not None:
        if isinstance(toas.filename, (str, Path)):
            picklefilename = f"{str(toas.filename)}.pickle.gz"
        else:
            picklefilename = f"{toas.filename[0]}.pickle.gz"
    else:
        raise ValueError("TOA pickle method needs a (single) filename.")
    with gzip.open(picklefilename, "wb") as f:
        pickle.dump(toas, f)


def get_TOAs_list(
    toa_list,
    ephem=None,
    include_bipm=True,
    bipm_version=bipm_default,
    include_gps=True,
    planets=False,
    tdb_method="default",
    commands=None,
    filename=None,
    hashes=None,
    limits="warn",
):
    """Load TOAs from a list of TOA objects.

    See :func:`pint.toa.get_TOAs` for details of what this function does.
    """
    t = TOAs(toalist=toa_list)
    t.commands = [] if commands is None else commands
    t.filename = filename
    t.hashes = {} if hashes is None else hashes
    if all("clkcorr" not in f for f in t.table["flags"]):
        t.apply_clock_corrections(
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            limits=limits,
        )
    if "tdb" not in t.table.colnames:
        t.compute_TDBs(method=tdb_method, ephem=ephem)
    if "ssb_obs_pos" not in t.table.colnames:
        t.compute_posvels(ephem, planets)
    return t


def _toa_format(line, fmt="Unknown"):
    """Determine the type of a TOA line.

    Identifies a TOA line as one of the following types:
    Comment, Command, Blank, Tempo2, Princeton, ITOA, Parkes, Unknown.
    """
    if re.match(r"[0-9a-z@] ", line):
        return "Princeton"
    elif (
        line.startswith("C ")
        or line.startswith("c ")  # FIXME: This matches the re above!
        or line.startswith("#")
        or line.startswith("CC ")
    ):
        return "Comment"
    elif line.upper().lstrip().startswith(toa_commands):
        return "Command"
    elif re.match(r"^\s*$", line):  # FIXME: what about empty lines?
        return "Blank"
    elif re.match(r"^ ", line) and len(line) > 41 and line[41] == ".":
        return "Parkes"
    elif len(line) > 80 or fmt == "Tempo2":
        return "Tempo2"
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == ".":
        # FIXME: This needs to be better
        return "ITOA"
    else:
        return "Unknown"


def _parse_TOA_line(line, fmt="Unknown"):
    """Parse a one-line ASCII time-of-arrival.

    Return an MJD tuple and a dictionary of other TOA information.
    The format can be one of: Comment, Command, Blank, Tempo2,
    Princeton, ITOA, Parkes, or Unknown.
    """
    MJD = None
    fmt = _toa_format(line, fmt)
    d = dict(format=fmt)
    if fmt == "Princeton":
        # Princeton format
        # ----------------
        # columns  item
        # 1-1     Observatory (one-character code) '@' is barycenter
        # 2-2     must be blank
        # 16-24   Observing frequency (MHz)
        # 25-44   TOA (decimal point must be in column 30 or column 31)
        # 45-53   TOA uncertainty (microseconds)
        # 69-78   DM correction (pc cm^-3)
        d["obs"] = get_observatory(line[0].upper()).name
        d["freq"] = float(line[15:24])
        d["error"] = float(line[44:53])
        ii, ff = line[24:44].split(".")
        ii = int(ii)
        # For very old TOAs, see https://tempo.sourceforge.net/ref_man_sections/toa.txt
        if ii < 40000:
            ii += 39126
        MJD = (ii, float(f"0.{ff}"))
        try:
            d["ddm"] = str(float(line[68:78]))
        except ValueError:
            d["ddm"] = str(0.0)
    elif fmt == "Tempo2":
        # This could use more error catching...
        fields = line.split()
        d["name"] = fields[0]
        d["freq"] = float(fields[1])
        if "." in fields[2]:
            ii, ff = fields[2].split(".")
            MJD = (int(ii), float(f"0.{ff}"))
        else:
            MJD = (int(fields[2]), 0.0)
        d["error"] = float(fields[3])
        d["obs"] = get_observatory(fields[4].upper()).name
        # All the rest should be flags
        flags = fields[5:]

        # Flags and flag-values should be given in pairs.
        # The for loop below will fail otherwise.
        if len(flags) % 2 != 0:
            raise ValueError(
                f"Flags and flag-values should be given in pairs. The given flags are {' '.join(flags)}"
            )

        for i in range(0, len(flags), 2):
            k, v = flags[i].lstrip("-"), flags[i + 1]
            if k in ["error", "freq", "scale", "MJD", "flags", "obs", "name"]:
                raise ValueError(f"TOA flag ({k}) will overwrite TOA parameter!")
            if not k:
                raise ValueError(f"The string {repr(flags[i])} is not a valid flag")
            d[k] = v
    elif fmt == "Command":
        d[fmt] = line.split()
    elif fmt == "Parkes":
        """
        columns     item
        1-1         Must be blank
        26-34       Observing Frequency (MHz)
        35-55       TOA (decimal point must be in column 42)
        56-63       Phase offset (fraction of P0, added to TOA)
        64-71       TOA uncertainty
        80-80       Observatory (1 character)
        """
        d["name"] = line[1:25]
        d["freq"] = float(line[25:34])
        ii = line[34:41]
        ff = line[42:55]
        MJD = int(ii), float(f"0.{ff}")
        phaseoffset = float(line[55:62])
        if phaseoffset != 0:
            raise ValueError(
                "Cannot interpret Parkes format with phaseoffset=%f yet" % phaseoffset
            )
        d["error"] = float(line[63:71])
        d["obs"] = get_observatory(line[79].upper()).name
    elif fmt == "ITOA":
        raise RuntimeError(f"TOA format '{fmt}' not implemented yet")
    elif fmt not in ["Blank", "Comment"]:
        raise RuntimeError(
            f"Unable to identify TOA format for line {line!r}, expecting {fmt}"
        )
    return MJD, d


def format_toa_line(
    toatime,
    toaerr,
    freq,
    obs,
    dm=0.0 * pint.dmu,
    name="unk",
    flags={},
    format="Princeton",
    alias_translation=None,
):
    """Format TOA line for writing

    Parameters
    ----------
    toatime : astropy.time.Time
        Time object containing TOA arrival time
    toaerr : astropy.units.Quantity
        TOA error as a Quantity with units
    freq : astropy.units.Quantity
        Frequency as a Quantity with units (NB: value of np.inf is allowed)
    obs : pint.observatory.Observatory
        Observatory object
    dm : astropy.units.Quantity
        DM for the TOA as a Quantity with units (not printed if 0.0 pc/cm^3)
    name : str
        Name to embed in TOA line (conventionally the data file name)
    format : str
        (Princeton | Tempo2)
    flags : dict
        Any Tempo2 flags to append to the TOA line
    alias_translation : dict or None
        Translate observatory names by looking them up in this dictionary;
        this may be necessary to convert observatory names into something
        TEMPO can understand, or to cope with different setups using
        different names for the same observatory or the same name
        for different observatories. There is a dictionary ``tempo_aliases``
        available to use names as compatible with TEMPO as possible.

    Returns
    -------
    out : str
        Formatted TOA line

    Note
    ----
    This implementation does not undo things like ``TIME`` statements; when used
    by :func:`pint.toa.TOAs.write_TOA_file` these commands are not emitted either.

    Princeton format::

        columns  item
        1-1     Observatory (one-character code) '@' is barycenter
        2-2     must be blank
        16-24   Observing frequency (MHz)
        25-44   TOA (decimal point must be in column 30 or column 31)
        45-53   TOA uncertainty (microseconds)
        69-78   DM correction (pc cm^-3)

    Tempo2 format:

        - First line of file should be "``FORMAT 1``"
        - TOA format is ``name freq sat satErr siteID <flags>``
    """
    if alias_translation is None:
        alias_translation = {}
    if format.upper() in ("TEMPO2", "1"):
        toa_str = Time(toatime, format="pulsar_mjd_string", scale=obs.timescale)
        # In Tempo2 format, freq=0.0 means infinite frequency
        if freq == np.inf * u.MHz:
            freq = 0.0 * u.MHz
        flagstring = ""
        if dm != 0.0 * pint.dmu:
            flagstring += "-dm {:.5f}".format(dm.to(pint.dmu).value)
        # Here I need to append any actual flags
        for flag in flags.keys():
            v = flags[flag]
            # Since toas file do not have values with unit in the flags,
            # here we are taking the units out
            if flag in ["clkcorr"]:
                continue
            if hasattr(v, "unit"):
                v = v.value
            flag = str(flag)
            flagstring += f" {flag} {v}" if flag.startswith("-") else f" -{flag} {v}"
        # Now set observatory code. Use obs.name unless overridden by tempo2_code
        try:
            obscode = obs.tempo2_code
        except AttributeError:
            obscode = obs.name
        out = "%s %f %s %.3f %s %s\n" % (
            name,
            freq.to(u.MHz).value,
            toa_str,
            toaerr.to(u.us).value,
            alias_translation.get(obscode, obscode),
            flagstring,
        )
    elif format.upper() in ("PRINCETON", "TEMPO"):
        # This should probably use obs.timescale instead of this hack
        if obs.tempo_code == "@":
            toa_str = str(Time(toatime, format="pulsar_mjd_string", scale="tdb"))
        else:
            toa_str = str(Time(toatime, format="pulsar_mjd_string", scale="utc"))
        # The Princeton format can only deal with MJDs that have up to 20
        # digits, so truncate if longer.
        if len(toa_str) > 20:
            toa_str = toa_str[:20]
        # In TEMPO/Princeton format, freq=0.0 means infinite frequency
        if freq == np.inf * u.MHz:
            freq = 0.0 * u.MHz
        if obs.tempo_code is None:
            raise ValueError(
                f"Observatory {obs.name} does not have 1-character tempo_code!"
            )
        if dm != 0.0 * pint.dmu:
            out = obs.tempo_code + " %13s%9.3f%20s%9.2f                %9.4f\n" % (
                name,
                freq.to(u.MHz).value,
                toa_str,
                toaerr.to(u.us).value,
                dm.to(pint.dmu).value,
            )
        else:
            out = obs.tempo_code + " %13s%9.3f%20s%9.2f\n" % (
                name,
                freq.to(u.MHz).value,
                toa_str,
                toaerr.to(u.us).value,
            )
    else:
        raise ValueError(f"Unknown TOA format ({format})")

    return out


def read_toa_file(filename, process_includes=True, cdict=None, dir=None):
    """Read TOAs from the given filename into a list.

    Will process INCLUDEd files unless process_includes is False.

    Parameters
    ----------
    filename : str or file-like object
        The name of the file to open, or an open file to read from.
    process_includes : bool, optional
        If true, obey INCLUDE directives in the file and read other
        files.
    top : bool, optional
        If true, wipe this instance's contents, otherwise append
        new TOAs. Used recursively; note that surprises may ensue
        if this function is called on an already existing and
        processed TOAs object.
    dir : str or Path
        This is the directory where the TOA file is found. INCLUDE
        statements are relative to this directory. If None, it is
        assumed to be the directory part of ``filename``, if that
        is a path rather than a file-like object. If None and ``filename``
        is a file-like object, the directory is assumed to be the
        current directory.
    """
    if isinstance(filename, (str, Path)):
        if dir is None:
            dir = Path(filename).parent
        with open(filename, "r") as f:
            return read_toa_file(
                f, process_includes=process_includes, cdict=cdict, dir=dir
            )
    else:
        f = filename
        if dir is None:
            dir = Path(".")

    ntoas = 0
    toas = []
    commands = []
    if cdict is None:
        cdict = {
            "EFAC": 1.0,
            "EQUAD": 0.0 * u.us,
            "EMIN": 0.0 * u.us,
            "EMAX": np.inf * u.us,
            "FMIN": 0.0 * u.MHz,
            "FMAX": np.inf * u.MHz,
            "INFO": None,
            "SKIP": False,
            "TIME": 0.0,
            "PHASE": 0,
            "PHA1": None,
            "PHA2": None,
            "MODE": 1,
            "JUMP": [False, 0],
            "FORMAT": "Unknown",
            "END": False,
        }
        top = True
    else:
        top = False
    for line in f.readlines():
        MJD, d = _parse_TOA_line(line, fmt=cdict["FORMAT"])
        if d["format"] == "Command":
            cmd = d["Command"][0].upper()
            commands.append((d["Command"], ntoas))
            if cmd == "SKIP":
                cdict[cmd] = True
                continue
            elif cmd == "NOSKIP":
                cdict["SKIP"] = False
                continue
            elif cmd == "END":
                cdict[cmd] = True
                break
            elif cmd in ("TIME", "PHASE"):
                cdict[cmd] += float(d["Command"][1])
            elif cmd in ("EMIN", "EMAX", "EQUAD"):
                cdict[cmd] = float(d["Command"][1]) * u.us
            elif cmd in ("FMIN", "FMAX", "EQUAD"):
                cdict[cmd] = float(d["Command"][1]) * u.MHz
            elif cmd in ("EFAC", "PHA1", "PHA2"):
                cdict[cmd] = float(d["Command"][1])
                if cmd in ("PHA1", "PHA2"):
                    d[cmd] = d["Command"][1]
            elif cmd == "INFO":
                cdict[cmd] = d["Command"][1]
                d[cmd] = d["Command"][1]
            elif cmd == "FORMAT":
                if d["Command"][1] == "1":
                    cdict[cmd] = "Tempo2"
            elif cmd == "JUMP":
                if cdict[cmd][0]:
                    cdict[cmd][0] = False
                    cdict[cmd][1] += 1
                else:
                    cdict[cmd][0] = True
            elif cmd == "INCLUDE" and process_includes:
                # Save FORMAT in a tmp
                fmt = cdict["FORMAT"]
                cdict["FORMAT"] = "Unknown"
                include_filename = Path(dir) / d["Command"][1]
                d["Command"][1] = str(include_filename)
                # Make filename relative to directory the parent file is in
                log.info(f"Processing included TOA file {include_filename}")
                new_toas, new_commands = read_toa_file(include_filename, cdict=cdict)
                toas.extend(new_toas)
                commands.extend(new_commands)
                # re-set FORMAT
                cdict["FORMAT"] = fmt
            elif cmd == "MODE":
                if d["Command"][1] == "0":
                    log.warning(
                        "MODE command is not supported by PINT. 'MODE 0' does not invoke ordinary least squares fitting."
                    )
                elif d["Command"][1] != "1":
                    log.warning("MODE command is not supported by PINT.")
                continue
            else:
                log.warning(f"Unknown command {cmd} in line {line}")
                continue
        if cdict["SKIP"] or d["format"] in ("Blank", "Unknown", "Comment", "Command"):
            continue
        elif cdict["END"]:
            if top:
                break
        else:
            newtoa = TOA(MJD, **d)
            if (
                (cdict["EMIN"] > newtoa.error)
                or (cdict["EMAX"] < newtoa.error)
                or (cdict["FMIN"] > newtoa.freq)
                or (cdict["FMAX"] < newtoa.freq)
            ):
                continue
            newtoa.error *= cdict["EFAC"]
            newtoa.error = np.hypot(newtoa.error, cdict["EQUAD"])
            if cdict["INFO"]:
                newtoa.flags["info"] = cdict["INFO"]
            if cdict["JUMP"][0]:
                newtoa.flags["jump"] = str(cdict["JUMP"][1] + 1)
                newtoa.flags["tim_jump"] = str(cdict["JUMP"][1] + 1)
            if cdict["PHASE"] != 0:
                newtoa.flags["phase"] = str(cdict["PHASE"])
            if cdict["TIME"] != 0.0:
                newtoa.flags["to"] = str(cdict["TIME"])
            toas.append(newtoa)
            ntoas += 1

    return toas, commands


def build_table(toas, filename=None):
    mjds, mjd_floats, errors, freqs, obss, flags = zip(
        *[
            (
                t.mjd,
                t.mjd.mjd,
                t.error.to_value(u.us),
                t.freq.to_value(u.MHz),
                t.obs,
                t.flags,
            )
            for t in toas
        ]
    )
    # np.array guesses the shape wrong for object arrays
    flags_array = np.empty(len(mjds), dtype=object)
    for i, f in enumerate(flags):
        flags_array[i] = f
    return table.Table(
        [
            np.arange(len(mjds)),
            table.Column(mjds),
            np.array(mjd_floats, dtype=float) * u.d,
            np.array(errors, dtype=float) * u.us,
            np.array(freqs, dtype=float) * u.MHz,
            np.array(obss),
            flags_array,
            np.zeros(len(mjds), dtype=float),
        ],
        names=(
            "index",
            "mjd",
            "mjd_float",
            "error",
            "freq",
            "obs",
            "flags",
            "delta_pulse_number",
        ),
        meta={"filename": filename},
    )


def _cluster_by_gaps(t, gap):
    """A utility function to cluster times according to gap-less stretches.

    This function is used by :func:`pint.toa.TOAs.get_clusters` to determine
    the clustering.

    Parameters
    ----------
    t : numpy.ndarray
        Input times to be clustered
    gap : float
        gap for clustering, same units as `t`

    Returns
    -------
    clusters : numpy.ndarray
        cluster numbers to which the times belong

    """
    ix = np.argsort(t)
    t_sorted = t[ix]
    gaps = np.diff(t_sorted)
    gap_starts = np.where(gaps >= gap)[0]
    gsi = np.concatenate(([0], gap_starts + 1, [len(t)]))
    clusters_sorted = np.repeat(np.arange(len(gap_starts) + 1), np.diff(gsi))
    clusters = np.zeros(len(t), dtype=int)
    clusters[ix] = clusters_sorted
    return clusters


class FlagDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = {}
        self.update(dict(*args, **kwargs))

    @staticmethod
    def from_dict(d):
        r = FlagDict()
        r.update(d)
        return r

    _key_re = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    @staticmethod
    def check_allowed_key(k):
        if not isinstance(k, str):
            raise ValueError(f"flag {k} must be a string")
        if k.startswith("-"):
            raise ValueError("flags should be stored without their leading -")
        if not FlagDict._key_re.match(k):
            raise ValueError(f"flag {k} is not a valid flag")

    @staticmethod
    def check_allowed_value(k, v):
        if not isinstance(v, str):
            raise ValueError(f"value {v} for key {k} must be a string")
        if not v and len(v.split()) != 1:
            raise ValueError(f"value {repr(v)} for key {k} cannot contain whitespace")

    def __setitem__(self, key, val):
        self.__class__.check_allowed_key(key)
        self.__class__.check_allowed_value(key, val)
        if val:
            self.store[key.lower()] = val
        elif key in self.store:
            del self.store[key]

    def __delitem__(self, key):
        del self.store[key.lower()]

    def __getitem__(self, key):
        return self.store[key.lower()]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f"FlagDict({repr(self.store)})"

    def __str__(self):
        return str(self.store)

    def copy(self):
        return FlagDict.from_dict(self.store)


class TOA:
    """A time of arrival (TOA) class.

    This is a class for representing a single pulse arrival
    time measurement. It carries both the time - which needs careful handling
    as we often need more precision than python floats can provide - and
    a collection of additional data necessary to work with the data. These
    are often obtained by reading ``.tim`` files produced by pulsar data
    analysis software, but they can also be constructed as python objects.

    Parameters
    ----------
    MJD : astropy.time.Time, float, or tuple of floats
        The time of the TOA, which can be expressed as an astropy Time,
        a floating point MJD (64 or 80 bit precision), or a tuple
        of (MJD1,MJD2) whose sum is the full precision MJD (usually the
        integer and fractional part of the MJD)
    error : astropy.units.Quantity or float
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds
    obs : str
        The observatory code for the TOA
    freq : float or astropy.units.Quantity
        Frequency corresponding to the TOA.  Either a Quantity with frequency
        units, or a number for which MHz is assumed.
    scale : str
        Time scale for the TOA time.  Defaults to the timescale appropriate
        to the site, but can be overridden
    flags : dict
        Flags associated with the TOA.  If flags is not provided, any
        additional keyword arguments are interpreted as flags.

    Attributes
    ----------
    mjd : astropy.time.Time
        The pulse arrival time
    error : astropy.units.Quantity
        The uncertainty on the pulse arrival time
    obs : str
        The observatory code
    freq : astropy.units.Quantity
        The observing frequency
    flags : dict
        Any additional flags that were set for this TOA

    Notes
    -----
    MJDs will be stored in astropy.time.Time format, and can be
    passed as a double (not recommended), a string, a
    tuple of component parts (usually day and fraction of day).
    error is the TOA uncertainty in microseconds
    obs is the observatory name as defined by the Observatory class
    freq is the observatory-centric frequency in MHz
    other keyword/value pairs can be specified as needed

    It is VERY important that all astropy.Time() objects are created
    with precision=9. This is ensured in the code and is checked for any
    Time object passed to the TOA constructor.

    A discussion of times and clock corrections in PINT is available here:
    https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

    Observatory codes are (semi-)standardized short strings describing
    particular observatories. PINT needs to know considerable additional
    information about the observatory, including its precise position and
    clock correction details.

    Examples
    --------

    Constructing a TOA object::

        >>> a = TOA((54567, 0.876876876876876), 4.5, freq=1400.0,
        ...         obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876:  4.500 us error from 'GBT' at 1400.0000 MHz {'backend': 'GUPPI'}

    """

    def __init__(
        self,
        MJD,
        error=0.0,
        obs="Barycenter",
        freq=float("inf"),
        scale=None,
        flags=None,
        **kwargs,
    ):
        site = get_observatory(obs)
        # If MJD is already a Time, just use it. Note that this will ignore
        # the 'scale' argument to the TOA() constructor!
        if isinstance(MJD, time.Time):
            if scale is not None:
                raise ValueError("scale argument is ignored when Time is provided")
            t = MJD
        else:
            try:
                arg1, arg2 = MJD
            except TypeError:
                arg1, arg2 = MJD, None
            if scale is None:
                scale = site.timescale
            # First build a time without a location
            # Note that when scale is UTC, must use pulsar_mjd format!
            fmt = "pulsar_mjd" if scale.lower() == "utc" else "mjd"
            t = time.Time(arg1, arg2, scale=scale, format=fmt, precision=9)

        # Now assign the site location to the Time, for use in the TDB conversion
        # Time objects are immutable so you must make a new one to add the location!
        # Use the initial time to look up the observatory location
        # (needed for moving observatories)
        # The location is an EarthLocation in the ITRF (ECEF, WGS84) frame
        try:
            loc = site.earth_location_itrf(time=t)
        except Exception:
            # Just add information and re-raise
            log.error(f"Error computing earth_location_itrf at time {t}, {type(t)}")
            raise
        # Then construct the full time, with observatory location set
        self.mjd = time.Time(t, location=loc, precision=9)

        if hasattr(error, "unit"):
            try:
                self.error = error.to(u.microsecond)
            except u.UnitConversionError:
                raise u.UnitConversionError(
                    f"Uncertainty for TOA with incompatible unit {error}"
                )
        else:
            self.error = error * u.microsecond
        self.obs = site.name
        if hasattr(freq, "unit"):
            try:
                self.freq = freq.to(u.MHz)
            except u.UnitConversionError as e:
                raise u.UnitConversionError(
                    f"Frequency for TOA with incompatible unit {freq}"
                ) from e
        else:
            self.freq = freq * u.MHz
        if self.freq == 0.0 * u.MHz:
            self.freq = np.inf * u.MHz
        if flags is None:
            self.flags = FlagDict.from_dict(kwargs)
        else:
            self.flags = FlagDict.from_dict(flags)
            if kwargs:
                raise TypeError(
                    f"TOA constructor does not accept keyword arguments {kwargs} when flags are specified."
                )

    def __str__(self):
        s = (
            self.mjd.mjd_string
            + f": {self.error.value:6.3f} {self.error.unit} error at '{self.obs}' at {self.freq.value:.4f} {self.freq.unit}"
        )
        if self.flags:
            s += f" {str(self.flags)}"
        return s

    def __eq__(self, other):
        result = True
        for p in ["mjd", "error", "obs", "freq", "flags"]:
            result = result and getattr(self, p) == getattr(other, p)
        return result

    def as_line(self, format="Tempo2", name=None, dm=0 * pint.dmu):
        """Format TOA as a line for a ``.tim`` file."""
        if name is not None:
            pass
        elif "name" in self.flags:
            name = self.flags["name"]
        else:
            raise ValueError("Unable to generate TOA line due to missing name.")
        return format_toa_line(
            toatime=self.mjd,
            toaerr=self.error,
            freq=self.freq,
            obs=get_observatory(self.obs),
            dm=dm,
            name=name,
            format=format,
            flags=self.flags,
        )


class TOAs:
    """A class of multiple TOAs, loaded from zero or more files.

    Normally these objects should be read from a file with :func:`pint.toa.get_TOAs`.
    Constructing them with the constructor here does not apply the clock
    corrections and the resulting TOAs object may not be set up the way one
    would normally expect.

    The contents are stored in an :class:`astropy.table.Table`. Not all columns of the table are computed automatically, as their
    computation can be expensive. Methods of this class are available to
    populate these additional columns. Methods that return data from the columns
    do so in the internal order.

    TOAs objects can accept indices that are boolean, list-of-integer, or
    slice, to produce a new TOAs object containing a subset of the TOAs in the
    original.  For example, to obtain a new TOAs object
    containing the entries above 1 GHz:

    >>> t[t.table['freq'] > 1*u.GHz]

    TOAs objects also accept indexing to select columns or flags, so that for example
    ``t['mjd']`` returns the :class:`~astropy.table.Column` contained in the object, and
    ``t['fish']`` returns an array of strings that has the value associated with the flag
    ``-fish`` for each TOA that has that flag or the empty string for each TOA that doesn't.
    TOAs objects also support assignment through these methods:

    >>> t['high', t['freq'] > 1*u.GHz] = "1"

    TOAs used to be grouped by observatory, but currently are not.  To iterate over all observatory groups:

    >>> for obs,idx in t.get_obs_groups():

    Or to actually reorder the TOAs by observatory:

    >>> t.table.group_by("obs")

    Other sorting is fine too:

    >>> t.table.sort("freq")


    .. list-table:: Columns in ``.table``
       :widths: 15 85
       :header-rows: 1

       * - Name
         - Contents
       * - ``index``
         - location of the TOA in the original input
       * - ``mjd``
         - the exact time of arrival (an :class:`astropy.time.Time` object)
       * - ``mjd_float``
         - the time of arrival in floating-point (may be microseconds off)
       * - ``error``
         - the standard error (an :class:`astropy.units.Quantity` describing
           the claimed uncertainty on the pulse arrival time)
       * - ``freq``
         - the observing frequency (an :class:`astropy.units.Quantity`)
       * - ``obs``
         - the observatory at which the TOA was acquired (a
           :class:`pint.observatory.Observatory` object)
       * - ``flags``
         - free-form flags associated with the TOA (a dictionary mapping flag
           to value)
       * - ``tdb``
         - the pulse arrival time converted to TDB (but not barycentered, that is,
           not corrected for light travel time; an :class:`astropy.time.Time` object);
           computed by :func:`pint.toa.TOAs.compute_TDBs`
       * - ``tdbld``
         - a ``longdouble`` version of ``tdb`` for computational convenience
       * - ``ssb_obs_pos``, ``ssb_obs_vel``
         - position and velocity of the observatory at the time of the TOA; computed
           by :func:`pint.toa.TOAs.compute_posvels`
       * - ``ssb_obs_vel_ecl``
         - velocity of the observatory in ecliptic coordinates at the time of the TOA; computed
           by :func:`pint.toa.TOAs.add_vel_ecl`
       * - ``obs_sun_pos``, ``obs_jupiter_pos``, ``obs_saturn_pos``, ``obs_venus_pos``,
           ``obs_uranus_pos``, ``obs_neptune_pos``, ``obs_earth_pos``
         - position of various celestial objects at the time of the TOA; computed
           by :func:`pint.toa.TOAs.compute_posvels`
       * - ``pulse_number``
         - integer number of turns since a fiducial moment;
           optional; can be computed from a model with
           :func:`pint.toa.TOAs.compute_pulse_numbers` or extracted from the
           ``pn`` entry in ``flags`` with
           :func:`pint.toa.TOAs.phase_columns_from_flags`.
           If it is present for some TOAs but not all, missing values are filled with ``NaN``.
       * - ``delta_pulse_number``
         - number of turns to adjust pulse number by, compared to the model;
           ``PHASE`` statements in the ``.tim`` file or the ``padd`` entry in
           ``flags`` carry this information, and :func:`pint.toa.TOAs.phase_columns_from_flags`
           creates the column.

    Parameters
    ----------
    toafile : str, optional
        Filename to load TOAs from.
    toalist : list of TOA objects, optional
        The TOA objects this TOAs should contain.
    toatable : astropy.table.Table, optional
        An existing TOA table
    tzr : bool
        Whether the TOAs object corresponds to a TZR TOA

    Exactly one of these three parameters must be provided.

    Attributes
    ----------
    table : astropy.table.Table
        The data for all the TOAs.
    commands : list of str
        "Commands" that were written in the file; these will have affected
        how some or all TOAs were interpreted during loading.
    filename : str, optional
        The filename (if any) that the TOAs were loaded from.
    planets : bool
        Whether planetary Shapiro delay should be considered.
    ephem : object
        The Solar System ephemeris in use.
    clock_corr_info : dict
        Information about the clock correction chains in use.
    merged : bool
        If this object was merged from several files (and thus the filename of
        the first is not useful for referring to the whole object).
    hashes : dict
        A dictionary of hashes of the files this data was read from (if any).
        This is used by ``check_hashes()`` to verify whether the data on disk
        has changed so that the file can be re-read if necessary.
    was_pickled : bool
        Whether this file was loaded from a pickle.
    alias_translation : dict or None
        Translate observatory names by looking them up in this dictionary;
        this may be necessary to convert observatory names into something
        TEMPO can understand, or to cope with different setups using
        different names for the same observatory or the same name
        for different observatories. There is a dictionary ``tempo_aliases``
        available to use names as compatible with TEMPO as possible.
    wideband : bool
        Whether the TOAs also have wideband DM information
    tzr : bool
        Whether the TOAs object corresponds to a TZR TOA
    """

    def __init__(self, toafile=None, toalist=None, toatable=None, tzr=False):
        # First, just make an empty container
        self.commands = []
        self.filename = None
        self.planets = False
        self.ephem = None
        self.clock_corr_info = {}
        self.obliquity = None
        self.merged = False
        self.hashes = {}
        self.was_pickled = False
        self.alias_translation = None
        self.tzr = tzr

        if (toalist is not None) and (toafile is not None):
            raise ValueError("Cannot initialize TOAs from both file and list.")
        if (toalist is not None) and (toatable is not None):
            raise ValueError("Cannot initialize TOAs from both table and list.")
        if (toatable is not None) and (toafile is not None):
            raise ValueError("Cannot initialize TOAs from both file and table.")

        if toatable is None:
            if isinstance(toafile, (str, Path)):
                toalist, self.commands = read_toa_file(toafile)
                # Check to see if there were any INCLUDEs:
                inc_fns = [
                    x[0][1] for x in self.commands if x[0][0].upper() == "INCLUDE"
                ]
                self.filename = [toafile] + inc_fns if inc_fns else toafile
            elif toafile is not None:
                toalist, self.commands = read_toa_file(toafile)
                self.filename = None

            if toalist is None:
                raise ValueError("No TOAs found!")
            if not isinstance(toalist, (list, tuple)):
                raise ValueError("Trying to initialize TOAs from a non-list class")
            self.table = build_table(toalist, filename=self.filename)
        else:
            self.table = copy.deepcopy(toatable)
        self.max_index = len(self.table) - 1
        # Add pulse number column (if needed) or make PHASE adjustments
        try:
            self.phase_columns_from_flags()
        except ValueError:
            log.debug("No pulse number flags found in the TOAs")

        # We don't need this now that we have a table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        """Extract a subset of TOAs and/or a column/flag from each one.

        When selecting a column from ``self.table`` or a flag from ``self.table["flags"]``,
        pass its name as a string (not including the ``-`` if it's a flag). The result will
        be a :class:`~astropy.table.Column` if the string is the name of a column in ``self.table``
        or a newly allocated array of strings if the string is the name of a flag. If the
        flag is not set for some or all of the TOAs, this array will contain the empty string
        at the corresponding place.

        When selecting a subset of the TOAs, a list/array of indices will
        result in selecting those TOAs (though not necessarily in that order),
        a list/array of Booleans will result in selecting those TOAs for which
        the list/array has True, and a slice will select the corresponding
        slice of TOAs.

        Both a column and subset can be selected at once by forming a tuple, as in
        ``toas["fish", ::10]``; this will result in selecting that subset of the
        appropriate column. This mode will allow selection of individual elements
        (``toas["fish", 17]``), unlike all-column selection.

        Parameter
        ---------
        index : str or pair or list or array or slice or int
            How to choose what to select.

        Returns
        -------
        pint.toa.TOAs or numpy.ndarray or astropy.table.Column
            The selected part of the object.

        Note
        ----
        This function does not currently support extracting a single :class:`~pint.toa.TOA` object.
        To use integer indexing a column must be selected, or a table with length 1 will be returned.
        """
        column = None
        subset = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("Invalid indexing")
            a, b = index
            column, subset = (a, b) if isinstance(a, str) else (b, a)
        elif isinstance(index, str):
            column = index
        else:
            subset = index

        if column is None:
            if isinstance(index, np.ndarray) and index.dtype == bool:
                r = copy.deepcopy(self)
                r.table = r.table[index]
                return r
            elif (
                isinstance(index, np.ndarray)
                and index.dtype == np.int64
                or isinstance(index, list)
            ):
                r = copy.deepcopy(self)
                r.table = r.table[index]
                return r
            elif isinstance(index, slice):
                r = copy.deepcopy(self)
                r.table = r.table[index]
                return r
            elif isinstance(index, int):
                log.info(
                    "TOAs do not support extraction of single TOA objects.  Returning TOAs of length 1"
                )
                r = copy.deepcopy(self)
                r.table = r.table[[index]]
                return r
            else:
                raise ValueError(f"Unable to index TOAs with {index}")
        elif column in self.table.columns:
            return self.table[column] if subset is None else self.table[column, subset]
        else:
            r = []
            if subset is None:
                r.extend(f.get(column, "") for f in self.table["flags"])
            elif isinstance(subset, int):
                return self.table["flags"][subset].get(column, "")
            else:
                r.extend(f.get(column, "") for f in self.table["flags"][subset])
            # FIXME: what to do if length zero? How to ensure it's a string array even then?
            return np.array(r)

    def __setitem__(self, index, value):
        """Set values in this object.

        This can set specified values into columns/flags or subsets of the same.

        Columns/flags are specified by giving a string (without the initial ``-`` for a flag).

        Subsets can be specified by a list/array of indices, a list/array of booleans, a slice, or
        a single integer.

        Parameters
        ----------
        index : str or pair
            Which parts of the object to set.
        value
            What to set the values to. If a single value, will be "broadcast", otherwise should
            be an iterable of the same size as the selected subset and of appropriate type.

        Notes
        ----_
        This function does not currently support assignment of sets of TOAs to replace subsets of
        the TOAs in a :class:`~pint.toa.TOAs` object. Thus a column must always be selected.

        Because of the way flags are stored internally, looking up
        ``toas["a_flag"]`` has to produce a newly allocated array, and
        modifying it cannot affect the flags on the original object.
        Unfortunately ``toas["a_flag"][7] = "value"`` will appear to work but
        will do nothing. Use ``toas["a_flag", 7] = "value"`` instead.  Flag values will be converted to strings.
        """
        column = None
        subset = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("Invalid indexing")
            a, b = index
            column, subset = (a, b) if isinstance(a, str) else (b, a)
        elif isinstance(index, str):
            column = index
        else:
            subset = index

        if column is None:
            raise ValueError("Unable to assign sets of TOAs")
        elif column in self.table.columns:
            if subset is None:
                self.table[column] = value
            else:
                self.table[column][subset] = value
        elif np.isscalar(value):
            if subset is None:
                for f in self.table["flags"]:
                    if value:
                        f[column] = str(value)
                    else:
                        with contextlib.suppress(KeyError):
                            del f[column]
            elif isinstance(subset, int):
                f = self.table["flags"][subset]
                if value:
                    f[column] = str(value)
                else:
                    with contextlib.suppress(KeyError):
                        del f[column]
            else:
                for f in self.table["flags"][subset]:
                    if value:
                        f[column] = str(value)
                    else:
                        with contextlib.suppress(KeyError):
                            del f[column]
        else:
            if subset is None:
                subset = range(len(self))
            if len(subset) != len(value):
                raise ValueError(
                    "Length of flag values must be equal to length of TOA subset"
                )
            for i in subset:
                self[column, i] = str(value[i])

    def __repr__(self):
        return f"{len(self)} TOAs starting at MJD {self.first_MJD}"

    def __eq__(self, other):
        sd, od = self.__dict__.copy(), other.__dict__.copy()
        st = sd.pop("table")
        ot = od.pop("table")
        return sd == od and np.all(st == ot)

    def __add__(self, other):
        """Addition operator, allowing merging/concatenation of TOAs"""
        if isinstance(other, type(self)):
            return merge_TOAs([self, other])
        raise TypeError(f"Do not know how to add '{type(self)}' and '{type(other)}'")

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.merge(other)
            return self
        raise TypeError(f"Do not know how to add '{type(self)}' and '{type(other)}'")

    def __sub__(self, other):
        if isinstance(other, (u.Quantity, time.TimeDelta)):
            return self + (-1 * other)
        raise TypeError(
            f"Do not know how to subtract '{type(self)}' and '{type(other)}'"
        )

    def __setstate__(self, state):
        # Normal unpickling behaviour
        self.__dict__.update(state)
        if not hasattr(self, "max_index"):
            self.max_index = np.maximum.reduce(self.table["index"])

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        # need to explicitly add in the flags
        for i in range(len(self)):
            result.table[i]["flags"] = copy.deepcopy(self.table[i]["flags"])
        return result

    @property
    def ntoas(self):
        """The number of TOAs. Also available as len(toas)."""
        return len(self.table)

    @property
    def observatories(self):
        """The set of observatories in use by these TOAs."""
        return set(self.get_obss())

    @property
    def first_MJD(self):
        """The first MJD, in :class:`~astropy.time.Time` format."""
        return self.get_mjds(high_precision=True).min()

    @property
    def last_MJD(self):
        """The last MJD, in :class:`~astropy.time.Time` format."""
        return self.get_mjds(high_precision=True).max()

    @property
    def wideband(self):
        """Whether or not the data have wideband TOA values"""
        return self.is_wideband()

    def to_TOA_list(self, clkcorr=False):
        """Turn a :class:`pint.toa.TOAs` object into a list of :class:`pint.toa.TOA` objects

        This effectively undoes :func:`pint.toa.get_TOAs_list`, optionally undoing clock corrections too

        Parameters
        ----------
        clkcorr : bool, optional
            Whether or not to undo any clock corrections

        Returns
        -------
        list :
            Of :class:`pint.toa.TOA` objects
        """
        tl = []
        clkcorrs = self.get_flag_value("clkcorr", 0, float)[0] * u.s
        for i in range(len(self)):
            t = self.table["mjd"][i]
            f = self.table["flags"][i]
            if not clkcorr:
                t -= clkcorrs[i]
                if "clkcorr" in f:
                    del f["clkcorr"]
            tl.append(
                TOA(
                    MJD=t,
                    error=self.table["error"][i] * self.table["error"].unit,
                    obs=self.table["obs"][i],
                    freq=self.table["freq"][i] * self.table["freq"].unit,
                    flags=f,
                )
            )
        return tl

    def is_wideband(self):
        """Whether or not the data have wideband TOA values"""

        # there may be a more elegant way to do this
        dm_data, valid_data = self.get_flag_value("pp_dm", as_type=float)
        return len(valid_data) == len(self)

    def get_all_flags(self):
        """Return a list of all the flags used by any TOA."""
        flags = set()
        for f in self.table["flags"]:
            flags.update(f.keys())
        return flags

    def get_freqs(self):
        """Return a :class:`~astropy.units.Quantity` of the observing frequencies for the TOAs."""
        return self.table["freq"].quantity

    def get_mjds(self, high_precision=False):
        """Array of MJDs in the TOAs object.

        With high_precision is True
        Return an array of the astropy.times (UTC) of the TOAs

        With high_precision is False
        Return an array of toas in mjd as double precision floats

        WARNING: Depending on the situation, you may get MJDs in a
        different scales (e.g. UTC, TT, or TDB) or even a mixture
        of scales if some TOAs are barycentred and some are not (a
        perfectly valid situation when fitting both Fermi and radio TOAs)
        """
        if high_precision:
            return np.array(self.table["mjd"])
        else:
            return self.table["mjd_float"].quantity

    def get_errors(self):
        """Return a numpy array of the TOA errors in us."""
        return self.table["error"].quantity

    def get_obss(self):
        """Return a numpy array of the observatories for each TOA."""
        return self.table["obs"]

    def get_obs_groups(self):
        """Return an iterator over the different observatories"""
        return pint.utils.group_iterator(self["obs"])

    def get_pulse_numbers(self):
        """Return a numpy array of the pulse numbers for each TOA if they exist."""
        # TODO: use a masked array?  Only some pulse numbers may be known
        if "pn" in self.table["flags"][0]:
            if "pulse_number" in self.table.colnames:
                raise ValueError("Pulse number cannot be both a column and a TOA flag")
            return np.array(self.get_flag_value("pn", np.nan, float)[0])
        elif "pulse_number" in self.table.colnames:
            return self.table["pulse_number"]
        else:
            log.warning("No pulse numbers for TOAs")
            return None

    def get_flags(self):
        """Return a numpy array of the TOA flags."""
        return self.table["flags"]

    def get_flag_value(self, flag, fill_value=None, as_type=None):
        """Get the requested TOA flag values.

        Parameters
        ----------
        flag_name : str
            The request flag name.
        fill_value
            The value to include for missing flags.
        as_type : callable or None
            If provided, this is called on each value to convert them
            from to the desired type. All flag values are stored as
            strings internally.

        Returns
        -------
        values : list
            A list of flag values from each TOA. If the TOA does not have
            the flag, it will fill up with the fill_value.
        valid_index : list
            The indices, in ``self.table``, of the places where the flag values occur.
        """
        result = []
        valid_index = []
        for ii, flags in enumerate(self.table["flags"]):
            try:
                val = flags[flag]
                valid_index.append(ii)
            except KeyError:
                val = fill_value
            else:
                if as_type is not None:
                    val = as_type(val)
            result.append(val)
        return result, valid_index

    def get_dms(self):
        """Get the Wideband DM data.

        Note
        ----
        This does not handle situations where some but not all TOAs have
        DM information.
        """
        result, valid = self.get_flag_value("pp_dm", as_type=float)
        if valid == []:
            raise AttributeError("No DM is provided.")
        return np.array(result)[valid] * pint.dmu

    def get_dm_errors(self):
        """Get the Wideband DM data error.

        Note
        ----
        This does not handle situations where some but not all TOAs have
        DM information.
        """
        result, valid = self.get_flag_value("pp_dme", as_type=float)
        if valid == []:
            raise AttributeError("No DM error is provided.")
        return np.array(result)[valid] * pint.dmu

    def get_clusters(self, gap_limit=2 * u.h, add_column=False, add_flag=None):
        """Identify toas within gap limit (default 2h = 0.0833d)
        of each other as the same cluster.

        Clusters can be larger than the gap limit - if toas are
        separated by a gap larger than the gap limit, a new cluster
        starts and continues until another such gap is found.

        Cluster info can be added as a ``clusters`` column to the
        :attr:`pint.toa.TOAs.table` object if `add_column` is True.  Cluster info can also be added as a flag with name specified.
        In those cases  ``self.table.meta['cluster_gap']``  will be set to the
        `gap_limit`.  If the desired clustering corresponds to that in
        :attr:`pint.toa.TOAs.table` then that column is returned.

        Parameters
        ----------
        gap_limit : astropy.units.Quantity, optional
            The minimum size of gap to create a new cluster. Defaults to two hours.
        add_column : bool, optional
            Whether or not to add a ``clusters`` column to the TOA table (default: False)
        add_flag : str, optional
            If not ``None``, will add a flag with that name to the TOA table whose value is the cluster number (as a string, starting at 0) (default: None)

        Returns
        -------
        clusters : numpy.ndarray
            The cluster number associated to each TOA. Clusters are numbered
            chronologically from zero.
        """
        if (
            "clusters" in self.table.colnames
            and "cluster_gap" in self.table.meta
            and gap_limit == self.table.meta["cluster_gap"]
        ):
            return self.table["clusters"]
        clusters = _cluster_by_gaps(
            self.get_mjds().to_value(u.d), gap_limit.to_value(u.d)
        )
        if add_column:
            self.table.add_column(clusters, name="clusters")
            self.table.meta["cluster_gap"] = gap_limit
            log.debug(f"Added 'clusters' column to TOA table with gap={gap_limit}")
        if add_flag is not None:
            for i in range(len(clusters)):
                self.table["flags"][i][add_flag] = str(clusters[i])
            self.table.meta["cluster_gap"] = gap_limit
            log.debug(f"Added '{add_flag}' flag to TOA table with gap={gap_limit}")

        return clusters

    def get_highest_density_range(self, ndays=7 * u.d):
        """Print the range of mjds (default 7 days) with the most toas"""
        sorted_mjds = np.sort(self.get_mjds())
        s = np.searchsorted(sorted_mjds, sorted_mjds + ndays)
        i = np.argmax(s - np.arange(len(sorted_mjds)))
        print(
            f"max density range is from MJD {sorted_mjds[i]} to {sorted_mjds[s[i]]} with {s[i]-i} TOAs."
        )
        return sorted_mjds[i], sorted_mjds[s[i]]

    def check_hashes(self, timfile=None):
        """Determine whether the input files are the same as when loaded.

        Parameters
        ----------
        timfile : str or list of str or file-like or None
            If provided this should match the list of files the TOAs object was loaded from.
            If this is a string or list of strings, and the number matches the number of
            files this TOAs object was loaded from, it is assumed that these are supposed to
            be the same files, re-named or moved; their contents are then checked. If the
            contents or the number doesn't match, this function returns False.

        Returns
        -------
        bool
            True if the contents of the TOAs object matches the content of the files.
        """
        if self.filename is None:
            return True
        elif isinstance(self.filename, (str, Path)):
            filenames = [self.filename]
        else:
            filenames = self.filename

        if timfile is None:
            timfiles = filenames
        elif hasattr(timfile, "readlines"):
            return True
        elif isinstance(timfile, (str, Path)):
            timfiles = [timfile]
        else:
            timfiles = list(timfile)
        if len(timfiles) != len(filenames):
            return False

        return all(
            pint.utils.compute_hash(t) == self.hashes[f]
            for t, f in zip(timfiles, filenames)
        )

    def select(self, selectarray):
        """Apply a boolean selection or mask array to the TOA table.

        Deprecated. Use ``toas[selectarray]`` to get a new object instead.

        This operation modifies the TOAs object in place, shrinking its
        table down to just those TOAs where selectarray is True. This
        function also stores the old table in a stack.
        """
        warnings.warn(
            "Please use boolean indexing on the object instead: toas[selectarray].",
            DeprecationWarning,
        )
        if hasattr(self, "table"):
            # Allow for selection undos
            if not hasattr(self, "table_selects"):
                self.table_selects = []
            self.table_selects.append(copy.deepcopy(self.table))
            self.table = self.table[selectarray]
        else:
            raise ValueError("TOA selection not implemented for TOA lists.")

    def unselect(self):
        """Return to previous selected version of the TOA table (stored in stack).

        Deprecated. Use ``toas[selectarray]`` to get a new object instead.
        """
        warnings.warn(
            "Please use boolean indexing on the object instead: toas[selectarray].",
            DeprecationWarning,
        )
        try:
            self.table = self.table_selects.pop()
        except (AttributeError, IndexError):
            log.error("No previous TOA table found.  No changes made.")

    def get_summary(self):
        """Return a short ASCII summary of the TOAs.

        This includes summary information about the errors and frequencies
        but is never more than a few lines regardless of how many TOAs there are.
        """
        s = f"Number of TOAs:  {self.ntoas}\n"
        if len(self.commands) and type(self.commands[0]) is list:
            s += f"Number of commands:  {[len(x) for x in self.commands]}\n"
        else:
            s += f"Number of commands:  {len(self.commands)}\n"
        s += f"Number of observatories: {len(self.observatories)} {list(self.observatories)}\n"
        s += f"MJD span:  {self.first_MJD.mjd:.3f} to {self.last_MJD.mjd:.3f}\n"
        s += f"Date span: {self.first_MJD.iso} to {self.last_MJD.iso}\n"
        for obs, grp in self.get_obs_groups():
            s += f"{obs} TOAs ({len(grp)}):\n"
            s += f"  Min freq:      {np.min(self['freq'][grp].to(u.MHz)):.3f}\n"
            s += f"  Max freq:      {np.max(self['freq'][grp].to(u.MHz)):.3f}\n"
            s += f"  Min error:     {np.min(self['error'][grp].to(u.us)):.3g}\n"
            s += f"  Max error:     {np.max(self['error'][grp].to(u.us)):.3g}\n"
            s += f"  Median error:  {np.median(self['error'][grp].to(u.us)):.3g}\n"
        return s

    def print_summary(self):
        """Prints self.get_summary()."""
        # FIXME: really do we need to have this function?
        print(self.get_summary())

    def phase_columns_from_flags(self):
        """Create and/or modify pulse_number and delta_pulse_number columns.

        Scans pulse numbers from the table flags and creates a new table column.
        Modifes the ``delta_pulse_number`` column, if required.
        Removes the pulse numbers from the flags.
        """
        # First get any PHASE commands
        dphs = np.array(self.get_flag_value("phase", 0, float)[0], dtype=np.float64)
        # Then add any -padd flag values
        dphs += np.array(self.get_flag_value("padd", 0, float)[0], dtype=np.float64)
        self.table["delta_pulse_number"] += dphs

        # Then, add pulse_number as a table column if possible
        pns = np.array(self.get_flag_value("pn", np.nan, float)[0])
        if np.all(np.isnan(pns)):
            raise ValueError("No pulse numbers found")
        self.table["pulse_number"] = pns
        self.table["pulse_number"].unit = u.dimensionless_unscaled

        # Remove pn from dictionary to prevent redundancies
        self["pn"] = ""
        # same for padd
        self["padd"] = ""

    def compute_pulse_numbers(self, model):
        """Set pulse numbers (in TOA table column pulse_numbers) based on model.

        Replace any existing pulse numbers by computing phases according to
        model and then setting the pulse number of each to their integer part,
        which the nearest integer since Phase objects ensure that.

        Parameters
        ----------
        model : pint.models.timing_model.TimingModel
            The model defining times of arrival; the pulse numbers assigned will
            be the nearest integer number of turns to that predicted by the model.
        """
        # paulr: I think pulse numbers should be computed with abs_phase=True!
        delta_pulse_numbers = Phase(self.table["delta_pulse_number"])
        phases = model.phase(self, abs_phase=True) + delta_pulse_numbers
        self.table["pulse_number"] = phases.int
        self.table["pulse_number"].unit = u.dimensionless_unscaled

    def remove_pulse_numbers(self):
        """Delete pulse numbers from TOA data"""

        if "pulse_number" in self.table.colnames:
            del self.table["pulse_number"]
        else:
            log.warning("Requested deleting of pulse numbers, but they are not present")

    def adjust_TOAs(self, delta):
        """Apply a time delta to TOAs.

        Adjusts the time (MJD) of the TOAs by applying delta, which should
        be a scalar or have the same shape as ``self.table['mjd']``.  This function does not change
        the pulse numbers column, if present, but does recompute ``mjd_float``,
        the TDB times, and the observatory positions and velocities.

        Parameters
        ----------
        delta : astropy.time.TimeDelta or astropy.units.Quantity
            The time difference to add to the MJD of each TOA
        """
        col = self.table["mjd"]
        if not isinstance(delta, (time.TimeDelta, u.Quantity)):
            raise ValueError("Type of argument must be Quantity or TimeDelta")
        if not isinstance(delta, time.TimeDelta):
            delta = time.TimeDelta(delta)
        if delta.isscalar:
            delta = time.TimeDelta(np.repeat(delta.sec, len(col)) * u.s)
        if delta.shape != col.shape:
            raise ValueError("Shape of mjd column and delta must be compatible")
        for ii in range(len(col)):
            col[ii] = col[ii] + delta[ii]

        # This adjustment invalidates the derived columns in the table, so delete
        # and recompute them
        # Changed high_precision from False to True to avoid self referential get_mjds()
        self.table["mjd_float"] = [
            t.mjd for t in self.get_mjds(high_precision=True)
        ] * u.day
        self.compute_TDBs()
        self.compute_posvels(self.ephem, self.planets)
        if self.obliquity is not None:
            self.add_vel_ecl(self.obliquity)

    def renumber(self, index_order=True):
        """Recreate the index column so the values go from 0 to len(self)-1.

        This modifies the TOAs object and also returns it, for calling
        convenience.

        Parameters
        ==========
        index_order : bool
            If True, preserve the order of the index column, but renumber so
            there are no gaps. If False, number according to the order TOAs
            occur in the object.

        Returns
        =======
        self
        """
        if index_order:
            ix = np.argsort(self.table["index"])
            self.table["index"][ix] = np.arange(len(self))
        else:
            self.table["index"] = np.arange(len(self))
        self.max_index = len(self) - 1
        return self

    def write_TOA_file(
        self,
        filename,
        name="unk",
        format="tempo2",
        commentflag=None,
        order_by_index=True,
        *,
        include_pn=True,
        include_info=True,
        comment=None,
    ):
        """Write this object to a ``.tim`` file.

        This function writes the contents of this object to a (single) ``.tim``
        file. If ``TEMPO2`` format is used, this file is able to represent the
        contents of this object to nanosecond level. No ``TIME`` or ``EFAC``
        commands are emitted.

        Parameters
        ----------
        filename : str or file-like
            File name to write to; can be an open file object
        name : str
            Value to put in the "name" field of tempo2 files, if a "-name" flag is
            not available.
        format : str
            Format specifier for file ('TEMPO' or 'Princeton') or ('Tempo2' or '1');
            note that not all features may be supported in 'TEMPO' mode.
        commentflag : str or None
            If a string, and that string is a TOA flag, that TOA will be commented
            in the output file.  If None (or non-string), no TOAs will be commented.
        order_by_index : bool
            If True, write the TOAs in the order specified in the "index" column
            (which is usually the same as the original file);
            if False, write them in the order they occur in the TOAs object.
        include_pn : bool, optional
            Include pulse numbers (if available)
        include_info : bool, optional
            Include information string if True
        comment : str, optional
            Additional string to include in TOA file
        """
        try:
            # FIXME: file must be closed even if an exception occurs!
            # Answer is to use a with statement and call the function recursively
            outf = open(filename, "w")
            handle = False
        except TypeError:
            outf = filename
            handle = True
        if format.upper() in ("TEMPO2", "1"):
            outf.write("FORMAT 1\n")
        if include_info:
            info_string = pint.utils.info_string(prefix_string="C ", comment=comment)
            outf.write(info_string + "\n")

        # Add pulse numbers to flags temporarily if there is a pulse number column
        # do this on a copy so as to leave the TOAs in a useful state
        # FIXME: everywhere else the pulse number column is called pulse_number not pn
        toacopy = copy.deepcopy(self)
        if "pulse_number" in toacopy.table.colnames:
            if include_pn:
                toacopy["pn"] = toacopy.table["pulse_number"]
                # but delete those that are NaN
                for i in np.where(np.isnan(self["pulse_number"]))[0]:
                    del toacopy.table["flags"][i]["pn"]
            else:
                log.warning(
                    "'pulse_number' column exists but it is not being written out"
                )
        if (
            "delta_pulse_number" in toacopy.table.columns
            and (toacopy.table["delta_pulse_number"] != 0).any()
        ):
            toacopy["padd"] = toacopy.table["delta_pulse_number"]

        if order_by_index:
            ix = np.argsort(toacopy.table["index"])
        else:
            ix = np.arange(len(toacopy))
        for toatime, toaerr, freq, obs, flags in zip(
            toacopy.table["mjd"][ix],
            toacopy.table["error"][ix].quantity,
            toacopy.table["freq"][ix].quantity,
            toacopy.table["obs"][ix],
            toacopy.table["flags"][ix],
        ):
            obs_obj = Observatory.get(obs)

            flags = flags.copy()
            toatime_out = toatime
            if "clkcorr" in flags:
                toatime_out -= time.TimeDelta(float(flags["clkcorr"]) * u.s)
            out_str = (
                "C " if isinstance(commentflag, str) and (commentflag in flags) else ""
            )
            out_str += format_toa_line(
                toatime_out,
                toaerr,
                freq,
                obs_obj,
                name=flags.pop("name", name),
                flags=flags,
                format=format,
                alias_translation=toacopy.alias_translation,
            )
            outf.write(out_str)

        if not handle:
            outf.close()

    def apply_clock_corrections(
        self,
        include_bipm=True,
        bipm_version=bipm_default,
        include_gps=True,
        limits="warn",
    ):
        """Apply observatory clock corrections and TIME statments.

        Apply clock corrections to all the TOAs where corrections are
        available.  This routine actually changes the value of the TOA,
        although the correction is also listed as a new flag for the TOA
        called 'clkcorr' so that it can be reversed if necessary.  This
        routine also applies all 'TIME' commands (``-to`` flags) and
        treats them exactly as if they were a part of the observatory
        clock corrections.

        If the clock corrections have already been applied they will not
        be re-applied.

        Options to include GPS or BIPM clock corrections are set to True
        by default in order to give the most accurate clock corrections.

        A description of how PINT handles clock corrections and timescales is here:
        https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

        Parameters
        ----------
        include_bipm : bool
            Whether or not to include BIPM correction
        bipm_version : str
            BIPM version to use.  The format must be 'BIPMXXXX' where XXXX is a year.
        include_gps : bool
            Whether or not to include GPS corrections
        limits : "warn" or "error"
            What to do when encountering TOAs for which clock corrections are not available.
        """
        # First make sure that we haven't already applied clock corrections
        flags = self.table["flags"]
        if any("clkcorr" in f for f in flags):
            if any("clkcorr" not in f for f in flags):
                # FIXME: could apply clock corrections to just the ones that don't have any
                raise ValueError("Some TOAs have 'clkcorr' flag and some do not!")
            log.warning("Clock corrections already applied. Not re-applying.")
            return
        # An array of all the time corrections, one for each TOA
        log.debug(
            f"Applying clock corrections (include_gps = {include_gps}, include_bipm = {include_bipm})"
        )
        corrections = np.zeros(self.ntoas) * u.s
        # values of "-to" flags
        time_statements = self.get_flag_value("to", 0, float)[0] * u.s
        for obs, grp in self.get_obs_groups():
            site = get_observatory(
                obs,
                include_gps=include_gps,
                include_bipm=include_bipm,
                bipm_version=bipm_version,
            )
            clock_corrections = site.clock_corrections(
                time.Time(self["mjd"][grp]), limits=limits
            )
            corrections[grp] = time_statements[grp] + clock_corrections
            for jj in grp:
                self["mjd"][jj] += time.TimeDelta(corrections[jj])

                if corrections[jj] != 0:
                    flags[jj]["clkcorr"] = str(corrections[jj].to_value(u.s))
            self["mjd_float"][grp] += corrections[grp].to_value(u.d)
        # Update clock correction info
        self.clock_corr_info.update(
            {
                "include_bipm": include_bipm,
                "bipm_version": bipm_version,
                "include_gps": include_gps,
            }
        )

    def compute_TDBs(self, method="default", ephem=None):
        """Compute and add TDB and TDB long double columns to the TOA table.

        This routine creates new columns 'tdb' and 'tdbld' in a TOA table
        for TDB times, using the Observatory locations and IERS Earth
        rotation corrections for UT1.

        If these columns are already present, delete and replace them.

        Parameters
        ----------
        method : str
            Which method to use. See :func:`pint.observatory.Observatory.get_TDBs`
            for details.
        ephem : str or None
            Solar System ephemeris to use for the computation. If not specified
            use the value in ``self.ephem`` if available, else use ``pint.toa.EPHEM_default``
            If specified, replace ``self.ephem``.
        """
        log.debug("Computing TDB columns.")
        if "tdb" in self.table.colnames:
            log.debug("tdb column already exists. Deleting...")
            self.table.remove_column("tdb")
        if "tdbld" in self.table.colnames:
            log.debug("tdbld column already exists. Deleting...")
            self.table.remove_column("tdbld")

        if ephem is None:
            if self.ephem is None:
                log.warning(
                    f"No ephemeris provided to TOAs object or compute_TDBs. Using {EPHEM_default}"
                )
                ephem = EPHEM_default
            else:
                ephem = self.ephem
        elif (self.ephem is not None) and (ephem != self.ephem):
            log.error(
                f"Ephemeris provided to compute_TDBs {ephem} is different than TOAs object ephemeris {self.ephem}! Using TDB ephemeris."
            )
        self.ephem = ephem
        log.debug(f"Using EPHEM = {self.ephem} for TDB calculation.")
        # Compute in observatory groups
        tdbs = np.zeros_like(self.table["mjd"])
        for obs, grp in self.get_obs_groups():
            site = get_observatory(obs)
            if isinstance(site, TopoObs):
                # For TopoObs, it is safe to assume that all TOAs have same location
                # I think we should report to astropy that initializing
                # a Time from a list (or Column) of Times throws away the location information
                grpmjds = time.Time(
                    self.table[grp]["mjd"], location=self.table[grp]["mjd"][0].location
                )
            elif isinstance(site, SatelliteObs):
                # for satellites, the location does not matter
                grpmjds = time.Time(self.table[grp]["mjd"], location=None)
            else:
                # Grab locations for each TOA
                loclist = np.array([t.location for t in self.table[grp]["mjd"]])
                if loclist[0] is None:
                    grpmjds = time.Time(self.table[grp]["mjd"], location=None)
                else:
                    locs = EarthLocation(
                        x=loclist["x"] * u.m, y=loclist["y"] * u.m, z=loclist["z"] * u.m
                    )
                    grpmjds = time.Time(self.table[grp]["mjd"], location=locs)

            tdbs[grp] = site.get_TDBs(grpmjds, method=method, ephem=ephem)
        # Now add the new columns to the table
        col_tdb = table.Column(name="tdb", data=tdbs)
        col_tdbld = table.Column(name="tdbld", data=[t.tdb.mjd_long for t in tdbs])
        self.table.add_columns([col_tdb, col_tdbld])

    def compute_posvels(self, ephem=None, planets=None):
        """Compute positions and velocities of the observatories and Earth.

        Compute the positions and velocities of the observatory (wrt
        the Geocenter) and the center of the Earth (referenced to the
        SSB) for each TOA.  The JPL solar system ephemeris can be set
        using the 'ephem' parameter.  The positions and velocities are
        set with PosVel class instances which have astropy units.

        If the required columns already exist, they will be replaced.

        Parameters
        ----------
        ephem : str
            The Solar System ephemeris to use; if not specified, use the
            default ephemeris for the TOAs object. If specified, replace
            the TOAs object's ``ephem`` attribute with this value and do
            the computation.
        planets : bool
            Whether to compute positions for the Solar System planets. If
            not specified, use the value stored in ``self.planets``; if
            specified, set ``self.planets`` to this value.
        """
        if ephem is None:
            if self.ephem is None:
                log.warning(
                    "No ephemeris provided to TOAs object or compute_posvels. Using DE421"
                )
                ephem = "DE421"
            else:
                ephem = self.ephem
        elif (self.ephem is not None) and (ephem != self.ephem):
            log.error(
                f"Ephemeris provided to compute_posvels {ephem} is different than TOAs object ephemeris {self.ephem}! Using posvels ephemeris."
            )
        if planets is None:
            planets = self.planets
        # Record the choice of ephemeris and planets
        self.ephem = ephem
        self.planets = planets
        if planets:
            log.debug(
                f"Computing PosVels of observatories, Earth and planets, using {ephem}"
            )

        else:
            log.debug(f"Computing PosVels of observatories and Earth, using {ephem}")
        # Remove any existing columns
        cols_to_remove = ["ssb_obs_pos", "ssb_obs_vel", "obs_sun_pos"]
        for c in cols_to_remove:
            if c in self.table.colnames:
                log.debug(f"Column {c} already exists. Removing...")
                self.table.remove_column(c)
        for p in all_planets:
            name = f"obs_{p}_pos"
            if name in self.table.colnames:
                log.debug(f"Column {name} already exists. Removing...")
                self.table.remove_column(name)

        self.table.meta["ephem"] = ephem
        ssb_obs_pos = table.Column(
            name="ssb_obs_pos",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km,
            meta={"origin": "SSB", "obj": "OBS"},
        )
        ssb_obs_vel = table.Column(
            name="ssb_obs_vel",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km / u.s,
            meta={"origin": "SSB", "obj": "OBS"},
        )
        obs_sun_pos = table.Column(
            name="obs_sun_pos",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km,
            meta={"origin": "OBS", "obj": "SUN"},
        )
        if planets:
            plan_poss = {}
            for p in all_planets:
                name = f"obs_{p}_pos"
                plan_poss[name] = table.Column(
                    name=name,
                    data=np.zeros((self.ntoas, 3), dtype=np.float64),
                    unit=u.km,
                    meta={"origin": "OBS", "obj": p},
                )

        # Now step through in observatory groups
        for obs, grp in self.get_obs_groups():
            site = get_observatory(obs)
            tdb = time.Time(self.table[grp]["tdb"], precision=9)

            if isinstance(site, T2SpacecraftObs):
                ssb_obs = site.posvel(tdb, ephem, group=self.table[grp])
            else:
                ssb_obs = site.posvel(tdb, ephem)

            log.debug(f"SSB obs pos {ssb_obs.pos[:, 0]}")
            ssb_obs_pos[grp, :] = ssb_obs.pos.T.to(u.km)
            ssb_obs_vel[grp, :] = ssb_obs.vel.T.to(u.km / u.s)
            sun_obs = objPosVel_wrt_SSB("sun", tdb, ephem) - ssb_obs
            obs_sun_pos[grp, :] = sun_obs.pos.T.to(u.km)
            if planets:
                for p in all_planets:
                    name = f"obs_{p}_pos"
                    dest = p
                    pv = objPosVel_wrt_SSB(dest, tdb, ephem) - ssb_obs
                    plan_poss[name][grp, :] = pv.pos.T.to(u.km)
        cols_to_add = [ssb_obs_pos, ssb_obs_vel, obs_sun_pos]
        if planets:
            cols_to_add += plan_poss.values()
        log.debug("Adding columns " + " ".join([cc.name for cc in cols_to_add]))
        self.table.add_columns(cols_to_add)

    def add_vel_ecl(self, obliquity):
        """Compute and add a column to self.table with velocities in ecliptic coordinates.

        Called in barycentric_radio_freq() in AstrometryEcliptic (astrometry.py)
        if ssb_obs_vel_ecl column does not already exist.
        If compute_posvels() called again for a TOAs object (aka TOAs modified),
        deletes this column so that this function will be called again and
        velocities will be calculated with updated TOAs.
        """
        # Remove any existing columns
        col_to_remove = "ssb_obs_vel_ecl"
        if col_to_remove in self.table.colnames:
            self.table.remove_column(col_to_remove)

        ssb_obs_vel_ecl = table.Column(
            name="ssb_obs_vel_ecl",
            data=np.zeros((self.ntoas, 3), dtype=np.float64),
            unit=u.km / u.s,
            meta={"origin": "SSB", "obj": "OBS"},
        )

        self.obliquity = obliquity
        ephem = self.ephem
        # Now step through in observatory groups
        for obs, grp in self.get_obs_groups():
            site = get_observatory(obs)
            tdb = time.Time(self["tdb"][grp], precision=9)
            if isinstance(site, T2SpacecraftObs):
                ssb_obs = site.posvel(tdb, ephem, self.table[grp])
            else:
                ssb_obs = site.posvel(tdb, ephem)

            # convert ssb_obs pos and vel to ecliptic coordinates
            coord = ICRS(
                x=ssb_obs.pos[0],
                y=ssb_obs.pos[1],
                z=ssb_obs.pos[2],
                v_x=ssb_obs.vel[0],
                v_y=ssb_obs.vel[1],
                v_z=ssb_obs.vel[2],
                representation_type=CartesianRepresentation,
                differential_type=CartesianDifferential,
            )
            coord = coord.transform_to(PulsarEcliptic(obliquity=obliquity))
            # get velocity vector from coordinate frame
            ssb_obs_vel_ecl[grp, :] = coord.velocity.d_xyz.T.to(u.km / u.s)
        col = ssb_obs_vel_ecl
        log.debug(f"Adding column {col.name}")
        self.table.add_column(col)

    def update_mjd_float(self):
        """Update the ``mjd_float`` column from the ``mjd`` column"""
        self["mjd_float"] = np.array([t.mjd for t in self["mjd"]], dtype=float) * u.d

    def update_all_times(self, tdb_method="default"):
        """Update the various derived time columns

        Updates:

            * ``mjd_float``
            * ``tdb``
            * ``tdbld``
            * ``ssb_obs_pos``
            * ``ssb_obs_vel``
            * ``obs_sun_pos``
            * ``obs_*_pos`` if ``planets`` is ``True``
            * ``ssb_obs_vel_ecl`` if ``obliquity`` is not ``None``

        Parameters
        ----------
        tdb_method : str
            Which method to use for the clock correction to TDB. See
            :func:`pint.observatory.Observatory.get_TDBs` for details.
        """
        self.compute_TDBs(method=tdb_method, ephem=self.ephem)
        self.compute_posvels(self.ephem, self.planets)
        self.update_mjd_float()

        if self.obliquity is not None:
            self.add_vel_ecl(self.obliquity)

    def merge(self, t, *args, strict=False):
        """Merge TOAs instances into the existing object

        In order for a merge to work, each TOAs instance needs to have
        been created using the same Solar System Ephemeris (EPHEM),
        the same reference timescale (i.e. CLOCK), the same value of
        .planets (i.e. whether planetary PosVel columns are in the tables
        or not), and the same value of the obliquity.

        If ``pulse_number``, [``ssb_obs_pos``, ``ssb_obs_vel``, ``obs_sun_pos``], [``tdb``, ``tdbld``]
        columns are present in some but not all data try to put them in unless ``strict=True``

        Parameters
        ----------
        t : pint.toa.TOAs
        args : pint.toa.TOAs, optional
            Additional TOAs to merge
        strict : bool, optional
            If ``strict``, do not add in missing columns to facilitate merging

        """
        TOAs_list = [self, t] + list(args)
        # Check each TOA object for consistency
        ephems = [tt.ephem for tt in TOAs_list]
        if len(set(ephems)) > 1:
            raise TypeError(f"merge_TOAs() cannot merge. Inconsistent ephem: {ephems}")
        inc_BIPM = [tt.clock_corr_info.get("include_bipm", None) for tt in TOAs_list]
        if len(set(inc_BIPM)) > 1:
            raise TypeError(
                f"merge_TOAs() cannot merge. Inconsistent include_bipm: {inc_BIPM}"
            )
        BIPM_vers = [tt.clock_corr_info.get("bipm_version", None) for tt in TOAs_list]
        if len(set(BIPM_vers)) > 1:
            raise TypeError(
                f"merge_TOAs() cannot merge. Inconsistent bipm_version: {BIPM_vers}"
            )
        inc_GPS = [tt.clock_corr_info.get("include_gps", None) for tt in TOAs_list]
        if len(set(inc_GPS)) > 1:
            raise TypeError(
                f"merge_TOAs() cannot merge. Inconsistent include_gps: {inc_GPS}"
            )
        planets = [tt.planets for tt in TOAs_list]
        if len(set(planets)) > 1:
            raise TypeError(
                f"merge_TOAs() cannot merge. Inconsistent planets: {planets}"
            )
        obliquity = [tt.obliquity for tt in TOAs_list]
        if len(set(obliquity)) > 1:
            raise TypeError(
                f"merge_TOAs() cannot merge. Inconsistent obliquity: {obliquity}"
            )

        # check for the presence of various computable columns
        #   pulse_number
        #   ssb_obs_pos, ssb_obs_vel, obs_sun_pos
        #   tdb, tdbld
        # if one file has these and the others don't, try to add them if strict=False
        all_colnames = [tt.table.colnames for tt in TOAs_list]
        has_pulse_number = np.array(
            ["pulse_number" in colnames for colnames in all_colnames]
        )
        has_posvel = np.array(
            [
                ("ssb_obs_pos" in colnames)
                and ("ssb_obs_vel" in colnames)
                and ("obs_sun_pos" in colnames)
                for colnames in all_colnames
            ]
        )
        has_tdb = np.array(
            [("tdb" in colnames) and ("tdbld" in colnames) for colnames in all_colnames]
        )
        has_posvel_ecl = np.array(
            ["ssb_obs_vel_ecl" in colnames for colnames in all_colnames]
        )
        if has_pulse_number.any() and not has_pulse_number.all():
            if strict:
                log.warning("Not all data have 'pulse_number' columns but strict=True")
            else:
                # some data have pulse_numbers but not all
                # put in NaN
                for i, tt in enumerate(TOAs_list):
                    if "pulse_number" not in tt.table.colnames:
                        log.warning(
                            f"'pulse_number' not present in data set {i}: inserting NaNs"
                        )
                        tt.table.add_column(
                            np.ones(len(tt)) * np.nan, name="pulse_number"
                        )
        if has_posvel.any() and not has_posvel.all():
            if strict:
                log.warning(
                    "Not all data have position/velocity columns but strict=True"
                )
            else:
                # some data have positions/velocities but not all
                # compute as needed
                for tt in TOAs_list:
                    if (
                        "ssb_obs_pos" not in tt.table.colnames
                        or "ssb_obs_vel" not in tt.table.colnames
                        or "obs_sun_pos" not in tt.table.colnames
                    ):
                        tt.compute_posvels()
        if has_posvel_ecl.any() and not has_posvel_ecl.all():
            if strict:
                log.warning(
                    "Not all data have ecliptic position/velocity columns but strict=True"
                )
            else:
                # some data have ecliptic positions/velocities but not all
                # compute as needed
                for tt in TOAs_list:
                    if "ssb_obs_vel_ecl" not in tt.table.colnames:
                        tt.add_vel_ecl(obliquity[0])
        if has_tdb.any() and not has_tdb.all():
            if strict:
                log.warning("Not all data have TDB columns but strict=True")
            else:
                # some data have TDBs but not all
                # compute as needed
                for tt in TOAs_list:
                    if (
                        "tdb" not in tt.table.colnames
                        or "tdbld" not in tt.table.colnames
                    ):
                        tt.compute_TDBs()

        # update in-place (in contrast to `merge_TOAs`)
        nt = self
        # The following ensures that the filename list is flat
        filenames = [copy.deepcopy(tt.filename) for tt in TOAs_list]
        nt.filename = []
        for xx in filenames:
            if type(xx) is list:
                nt.filename.extend(iter(xx))
            else:
                nt.filename.append(xx)
        # We do not ensure that the command list is flat
        nt.commands = [tt.commands for tt in TOAs_list]
        # Now do the actual table stacking
        start_index = 0
        tables = []
        for tt in TOAs_list:
            t = copy.deepcopy(tt.table)
            t["index"] += start_index
            start_index += tt.max_index + 1
            tables.append(t)
        try:
            nt.table = table.vstack(
                tables, join_type="exact", metadata_conflicts="silent"
            )
        except table.np_utils.TableMergeError as e:
            # the astropy vstack will enforce having the same columns throughout
            # but the error messages aren't very useful.
            # This isn't an exhaustive check
            # (it just checks everything against the first observation)
            # but it should be more helpful
            message = []
            for i, colnames in enumerate(all_colnames[1:]):
                extra_columns = [x for x in colnames if x not in all_colnames[0]]
                missing_columns = [x for x in all_colnames[0] if x not in colnames]
                if extra_columns:
                    message.append(
                        f"File {i+1} has extra column(s): {','.join(extra_columns)}"
                    )

                if missing_columns:
                    message.append(
                        f"File {i+1} has missing column(s): {','.join(missing_columns)}"
                    )
            raise table.np_utils.TableMergeError(", ".join(message)) from e

            # Fix the table meta data about filenames
        nt.table.meta["filename"] = nt.filename
        nt.max_index = start_index - 1
        for tt in TOAs_list[1:]:
            nt.hashes.update(tt.hashes)
        # This sets a flag that indicates that we have merged TOAs instances
        nt.merged = True


def merge_TOAs(TOAs_list, strict=False):
    """Merge a list of TOAs instances and return a new combined TOAs instance

    In order for a merge to work, each TOAs instance needs to have
    been created using the same Solar System Ephemeris (EPHEM),
    the same reference timescale (i.e. CLOCK), and the same value of
    .planets (i.e. whether planetary PosVel columns are in the tables
    or not).

    If ``pulse_number``, [``ssb_obs_pos``, ``ssb_obs_vel``, ``obs_sun_pos``], [``tdb``, ``tdbld``]
    columns are present in some but not all data try to put them in unless ``strict=True``

    Parameters
    ----------
    TOAs_list : list of TOAs instances
    strict : bool, optional
        If ``strict``, do not add in missing columns to facilitate merging

    Returns
    -------
    :class:`pint.toa.TOAs`
        A new TOAs instance with all the combined TOAs
    """
    # don't duplicate code: just use the existing method
    t = copy.deepcopy(TOAs_list[0])
    if len(TOAs_list) > 1:
        t.merge(*TOAs_list[1:], strict=strict)
    return t


def get_TOAs_array(
    times,
    obs,
    scale=None,
    errors=1 * u.us,
    freqs=np.inf * u.MHz,
    flags=None,
    model=None,
    ephem=None,
    include_bipm=True,
    bipm_version=bipm_default,
    include_gps=True,
    planets=False,
    tdb_method="default",
    commands=None,
    hashes=None,
    limits="warn",
    tzr=False,
    **kwargs,
):
    """Load and prepare TOAs for PINT use from an array of times.

    Creates TOAs from a an array of times, applies clock corrections, computes
    key values (like TDB), computes the observatory position and velocity
    vectors.

    Observatory clock corrections are also applied, and thus
    the exact result also depends on the precise values in observatory clock
    correction files; normally these do not change. See :func:`pint.toa.TOAs.apply_clock_corrections` f
    or further information on the meaning of
    the clock correction flags.

    Note that this requires all TOAs to be measured at a single observatory (which can be a
    spacecraft).  If you wish to combine multiple data-sets use :func:`pint.toa.merge_TOAs`

    Parameters
    ----------
    times : astropy.time.Time, float, numpy.ndarray, or tuple of floats/numpy.ndarray
        The times of the TOAs, which can be expressed as astropy Time values,
        floating point MJDs (64 or 80 bit precision), or a tuple
        of (MJD1,MJD2) values whose sum is the full precision MJD (usually the
        integer and fractional part of the MJD)
    obs : str
        The observatory code for the TOA
    errors : astropy.units.Quantity or numpy.ndarray or float, optional
        The uncertainty on the TOA; Assumed to be
        in microseconds if no units provided
    freqs : astropy.units.Quantity or numpy.ndarray or float, optional
        Frequency corresponding to the TOAs; Assumed to be
        in MHz if no units provided
    scale : str, optional
        Time scale for the TOAs.  Defaults to the timescale appropriate
        to the site, but can be overridden
    flags : dict or iterable
        Flags associated with the TOAs.  If iterable, then must have same length as ``times``.
        Any additional keyword arguments are interpreted as flags as well.
    ephem : str
        The name of the solar system ephemeris to use; defaults to "DE421".
    include_bipm : bool or None
        Whether to apply the BIPM clock correction. Defaults to True.
    bipm_version : str or None
        Which version of the BIPM tables to use for the clock correction.
        The format must be 'BIPMXXXX' where XXXX is a year.
    include_gps : bool or None
        Whether to include the GPS clock correction. Defaults to True.
    planets : bool or None
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.
    model : pint.models.timing_model.TimingModel or None
        If a valid timing model is passed, model commands (such as BIPM version,
        planet shapiro delay, and solar system ephemeris) that affect TOA loading
        are applied.
    tdb_method : str
        Which method to use for the clock correction to TDB. See
        :func:`pint.observatory.Observatory.get_TDBs` for details.
    hashes : dict
        A dictionary of hashes of the files this data was read from (if any).
        This is used by ``check_hashes()`` to verify whether the data on disk
        has changed so that the file can be re-read if necessary.
    limits : "warn" or "error"
        What to do when encountering TOAs for which clock corrections are not available.
    tzr : bool
        Whether the TOAs object corresponds to a TZR TOA

    Returns
    -------
    TOAs : pint.toa.TOAs
        Completed TOAs object representing the data.


    Examples
    --------

        >>> from pint import pulsar_mjd
        >>> from pint import toa
        >>> from astropy import time
        >>> t = pulsar_mjd.Time(np.linspace(55000, 56000, 100), scale="utc", format="pulsar_mjd")
        >>> errors = np.random.normal(loc=1, scale=0.1, size=len(t)) << u.us
        >>> freqs = np.ones(len(t)) * 1400 << u.MHz
        >>> obs = "gbt"
        >>> flags = {"quality": "good"}
        >>> # additional kwargs are converted to flags
        >>> # they are automatically made into strings
        >>> toas = toa.get_TOAs_array(
                t,
                obs,
                errors=errors,
                freqs=freqs,
                ephem="DE440",
                flags=flags,
                int_time=np.ones(len(t)) * 30,
            )
    """
    if model:
        # If the keyword args are set, override what is in the model
        if ephem is None and model["EPHEM"].value is not None:
            ephem = model["EPHEM"].value
            log.debug(f"Using EPHEM = {ephem} from the given model")
        if include_bipm is None and model["CLOCK"].value is not None:
            if model["CLOCK"].value == "TT(TAI)":
                include_bipm = False
                log.info("Using CLOCK = TT(TAI), so setting include_bipm = False")
            elif "BIPM" in model["CLOCK"].value:
                clk = model["CLOCK"].value.strip(")").split("(")
                if len(clk) == 2:
                    ctype, cvers = clk
                    if ctype == "TT" and cvers.startswith("BIPM"):
                        include_bipm = True
                        if bipm_version is None:
                            if cvers == "BIPM":
                                bipm_version = bipm_default
                            else:
                                bipm_version = cvers
                                log.debug(
                                    f"Using CLOCK = {bipm_version} from the given model"
                                )
                    else:
                        log.warning(
                            f'CLOCK = {model["CLOCK"].value} is not implemented. '
                            f"Using TT({bipm_default}) instead."
                        )
            else:
                log.warning(
                    f'CLOCK = {model["CLOCK"].value} is not implemented. '
                    f"Using TT({bipm_default}) instead."
                )
        if planets is None and model["PLANET_SHAPIRO"].value:
            planets = True
            log.debug("Using PLANET_SHAPIRO = True from the given model")

    # mjds, mjd_floats, errors, freqs, obss, flags
    site = get_observatory(obs)
    # If MJD is already a Time, just use it. Note that this will ignore
    # the 'scale' argument to the TOA() constructor!
    if isinstance(times, time.Time):
        if scale is not None and scale != times.scale:
            raise ValueError("scale argument is ignored when Time is provided")
        t = np.atleast_1d(times)
    else:
        arg1, arg2 = times if isinstance(times, tuple) else (times, None)
        arg1 = np.atleast_1d(arg1)
        arg2 = np.atleast_1d(arg2) if arg2 is not None else arg2
        if scale is None:
            scale = site.timescale
        # First build a time without a location
        # Note that when scale is UTC, must use pulsar_mjd format!
        fmt = "pulsar_mjd" if scale.lower() == "utc" else "mjd"
        t = time.Time(arg1, arg2, scale=scale, format=fmt, precision=9)
    # Now assign the site location to the Time, for use in the TDB conversion
    # Time objects are immutable so you must make a new one to add the location!
    # Use the intial time to look up the observatory location
    # (needed for moving observatories)
    # The location is an EarthLocation in the ITRF (ECEF, WGS84) frame
    try:
        loc = site.earth_location_itrf(time=t)
    except Exception:
        # Just add informmation and re-raise
        log.error(f"Error computing earth_location_itrf at time {t}, {type(t)}")
        raise
    # Then construct the full time, with observatory location set
    mjd = time.Time(t, location=loc, precision=9)

    errors = np.atleast_1d(errors)
    if len(errors) == 1:
        errors = np.repeat(errors, len(t))
    if len(errors) != len(t):
        raise AttributeError(
            f"Length of error array ({len(errors)}) must match length of times ({len(t)})"
        )
    if hasattr(errors, "unit"):
        try:
            errors = errors.to(u.microsecond)
        except u.UnitConversionError:
            raise u.UnitConversionError(
                f"Uncertainty for TOA with incompatible unit {errors}"
            )
    else:
        errors = errors * u.microsecond

    obs = [site.name] * len(t)

    freqs = np.atleast_1d(freqs)
    if len(freqs) == 1:
        freqs = np.repeat(freqs, len(t))
    if len(freqs) != len(t):
        raise AttributeError(
            f"Length of frequency array ({len(freqs)}) must match length of times ({len(t)})"
        )

    if hasattr(freqs, "unit"):
        try:
            freqs = freqs.to(u.MHz)
        except u.UnitConversionError:
            raise u.UnitConversionError(
                f"Frequency for TOA with incompatible unit {freqs}"
            )
    else:
        freqs = freqs * u.MHz
    freqs[freqs == 0] = np.inf * u.MHz

    if isinstance(flags, (list, tuple, np.ndarray)):
        if len(flags) != len(t):
            raise AttributeError(
                f"Length of flags ({len(flags)}) must match length of times ({len(t)})"
            )
        flagdicts = [FlagDict.from_dict(f) for f in flags]
    elif flags is not None:
        flagdicts = [FlagDict(flags) for i in range(len(t))]
    else:
        flagdicts = [FlagDict() for i in range(len(t))]

    for k, v in kwargs.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) != len(t):
                raise AttributeError(
                    f"Length of flags for {k} ({len(v)}) must match length of times ({len(t)})"
                )
            for i in range(len(v)):
                flagdicts[i][k] = str(v[i])
        else:
            for i in range(len(flagdicts)):
                flagdicts[i][k] = str(v)

    flags_array = np.empty(len(t), dtype=object)
    for i, f in enumerate(flagdicts):
        flags_array[i] = copy.deepcopy(f)

    out = table.Table(
        [
            np.arange(len(mjd)),
            table.Column(mjd),
            np.array(mjd.mjd, dtype=float) * u.d,
            np.array(errors, dtype=float) * u.us,
            np.array(freqs, dtype=float) * u.MHz,
            np.array(obs),
            flags_array,
            np.zeros(len(mjd), dtype=float),
        ],
        names=(
            "index",
            "mjd",
            "mjd_float",
            "error",
            "freq",
            "obs",
            "flags",
            "delta_pulse_number",
        ),
    )
    t = TOAs(toatable=out, tzr=tzr)
    t.commands = [] if commands is None else commands
    t.hashes = {} if hashes is None else hashes
    if all("clkcorr" not in f for f in t.table["flags"]):
        t.apply_clock_corrections(
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            limits=limits,
        )
    if "tdb" not in t.table.colnames:
        t.compute_TDBs(method=tdb_method, ephem=ephem)
    if "ssb_obs_pos" not in t.table.colnames:
        t.compute_posvels(ephem, planets)

    return t
