"""Generic functions to load TOAs from events files, along with specific implementations for different missions.
The versions that look like ``get_..._TOAs()`` are preferred: the others are retained for backward compatibility.

**Instrument-specific Functions**

.. autofunction:: pint.event_toas.get_NuSTAR_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.get_NICER_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.get_RXTE_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.get_IXPE_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.get_Swift_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.get_XMM_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.load_NuSTAR_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.load_NICER_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.load_RXTE_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.load_IXPE_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.load_Swift_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])
.. autofunction:: pint.event_toas.load_XMM_TOAs(eventname [, minmjd, maxmjd, errors, ephem, planets])

"""
import os
from functools import partial

import astropy.io.fits as pyfits
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u
import numpy as np
from loguru import logger as log

import pint.toa as toa
from pint.fits_utils import read_fits_event_mjds_tuples

"""
Default TOA (event) uncertainty depending on facility

* RXTE: https://ui.adsabs.harvard.edu/abs/1998ApJ...501..749R/abstract
* IXPE: https://ui.adsabs.harvard.edu/abs/2019SPIE11118E..0VO/abstract
* XMM: https://ui.adsabs.harvard.edu/abs/2012A%26A...545A.126M/abstract
* NuSTAR: https://ui.adsabs.harvard.edu/abs/2021ApJ...908..184B/abstract
* Swift: https://ui.adsabs.harvard.edu/abs/2005SPIE.5898..377C/abstract
* NICER: https://heasarc.gsfc.nasa.gov/docs/nicer/mission_guide/
"""
_default_uncertainty = {
    "NICER": 0.1 * u.us,
    "RXTE": 2.5 * u.us,
    "IXPE": 20 * u.us,
    "XMM": 48 * u.us,
    "NuSTAR": 65 * u.us,
    "Swift": 300 * u.us,
    "default": 1 * u.us,
}


__all__ = [
    "load_fits_TOAs",
    "load_event_TOAs",
    "load_NuSTAR_TOAs",
    "load_NICER_TOAs",
    "load_IXPE_TOAs",
    "load_RXTE_TOAs",
    "load_Swift_TOAs",
    "load_XMM_TOAs",
    "get_fits_TOAs",
    "get_event_TOAs",
    "get_NuSTAR_TOAs",
    "get_NICER_TOAs",
    "get_IXPE_TOAs",
    "get_RXTE_TOAs",
    "get_Swift_TOAs",
    "get_XMM_TOAs",
]


def read_mission_info_from_heasoft():
    """Read all the relevant information about missions in xselect.mdb."""

    if not os.getenv("HEADAS"):
        return {}

    fname = os.path.join(os.getenv("HEADAS"), "bin", "xselect.mdb")
    if not os.path.exists(fname):
        return {}

    db = {}
    with open(fname) as fobj:
        for line in fobj:
            line = line.strip()

            if line.startswith("!") or line == "":
                continue
            allvals = line.split()
            string = allvals[0]
            value = allvals[1:]
            if len(value) == 1:
                value = value[0]

            data = string.split(":")[:]
            mission = data[0].lower()
            if mission not in db:
                db[mission] = {}
            previous_db_step = db[mission]

            data = data[1:]
            for key in data[:-1]:
                if key not in previous_db_step:
                    previous_db_step[key] = {}
                previous_db_step = previous_db_step[key]
            previous_db_step[data[-1]] = value
    return db


# fits_extension can be a single name or a comma-separated list of allowed
# extension names.
# For weight we use the same conventions used for Fermi: None, a valid FITS
# extension name or CALC.
def create_mission_config():
    mission_config = {
        "generic": {
            "fits_extension": 1,
            "allow_local": False,
            "fits_columns": {"pi": "PI"},
        }
    }

    # Read the relevant information from $HEADAS/bin/xselect.mdb, if present
    for mission, data in read_mission_info_from_heasoft().items():
        mission_config[mission] = {"allow_local": False}
        ext = 1
        if "events" in data:
            ext = data["events"]
        cols = {}
        if "ecol" in data:
            ecol = data["ecol"]
            cols = {"ecol": str(ecol)}
        mission_config[mission]["fits_extension"] = ext
        mission_config[mission]["fits_columns"] = cols

    # Allow local TOAs for those missions that have a load_<MISSION>_TOAs method
    for mission in ["nustar", "nicer", "xte", "ixpe"]:
        try:
            # If mission was read from HEASARC, just update allow_local
            mission_config[mission]["allow_local"] = True
        except KeyError:
            # If mission was not read from HEASARC, then create full new entry
            mission_config[mission] = {}
            mission_config[mission].update(mission_config["generic"])
            mission_config[mission]["allow_local"] = True

    # Fix chandra
    try:
        mission_config["chandra"] = mission_config["axaf"]
    except KeyError:
        log.warning(
            "AXAF configuration not found -- likely HEADAS env variable not set."
        )

    # Fix xte
    mission_config["xte"]["fits_columns"] = {"ecol": "PHA"}
    # The event extension is called in different ways for different data modes, but
    # it is always no. 1.
    mission_config["xte"]["fits_extension"] = 1
    return mission_config


mission_config = create_mission_config()


def _default_obs_and_scale(mission, timesys, timeref):
    """Default values of observatory and scale, given TIMESYS and TIMEREF.

    In standard FITS files,
    + TIMESYS will be
        'TT' for unmodified events (or geocentered), and
        'TDB' for events barycentered with gtbary
    + TIMEREF will be
        'GEOCENTER' for geocentered events,
        'SOLARSYSTEM' for barycentered,
        'LOCAL' for unmodified events

    """
    if timesys == "TDB":
        log.info("Building barycentered TOAs")
        obs, scale = "Barycenter", "tdb"
    elif timeref == "LOCAL":
        log.info("Building spacecraft local TOAs")
        obs, scale = mission, "tt"
    else:
        log.info("Building geocentered TOAs")
        obs, scale = "Geocenter", "tt"

    return obs, scale


def _get_columns_from_fits(hdu, cols):
    new_dict = {}
    event_dat = hdu.data
    default_val = np.zeros(len(event_dat))
    # Parse and retrieve default values from the FITS columns listed in config
    for col in cols.keys():
        try:
            val = event_dat.field(cols[col])
        except ValueError:
            val = default_val
        new_dict[col] = val
    return new_dict


def _get_timesys_and_timeref(hdu):
    timesys, timeref = _get_timesys(hdu), _get_timeref(hdu)
    check_timesys(timesys)
    check_timeref(timeref)
    return timesys, timeref


VALID_TIMESYS = ["TDB", "TT"]
VALID_TIMEREF = ["GEOCENTER", "SOLARSYSTEM", "LOCAL"]


def check_timesys(timesys):
    if timesys not in VALID_TIMESYS:
        raise ValueError("Timesys has to be TDB or TT")


def check_timeref(timeref):
    if timeref not in VALID_TIMEREF:
        raise ValueError("Timeref is invalid")


def _get_timesys(hdu):
    event_hdr = hdu.header
    timesys = event_hdr["TIMESYS"]
    log.debug("TIMESYS {0}".format(timesys))
    return timesys


def _get_timeref(hdu):
    event_hdr = hdu.header

    timeref = event_hdr["TIMEREF"]
    log.debug("TIMEREF {0}".format(timeref))
    return timeref


def load_fits_TOAs(
    eventname,
    mission,
    weights=None,
    extension=None,
    timesys=None,
    timeref=None,
    minmjd=-np.inf,
    maxmjd=np.inf,
    errors=_default_uncertainty,
):
    """
    Read photon event times out of a FITS file as a list of PINT :class:`~pint.toa.TOA` objects.

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered TOAs. Different conditions may apply to different missions.

    The minmjd/maxmjd parameters can be used to avoid instantiation of TOAs
    we don't want, which can otherwise be very slow.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    mission : str
        Name of the mission (e.g. RXTE, XMM)
    weights : array or None
        The array has to be of the same size as the event list. Overwrites
        possible weight lists from mission-specific FITS files
    extension : str
        FITS extension to read
    timesys : str, default None
        Force this time system
    timeref : str, default None
        Forse this time reference
    minmjd : float, default "-infinity"
        minimum MJD timestamp to return
    maxmjd : float, default "infinity"
        maximum MJD timestamp to return
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    toalist : list of :class:`~pint.toa.TOA` objects

    Note
    ----
    This list should be converted into a :class:`~pint.toa.TOAs` object with :func:`pint.toa.get_TOAs_list` for most operations

    See Also
    --------
    :func:`get_fits_TOAs`
    """
    toas = get_fits_TOAs(
        eventname,
        mission,
        weights=weights,
        extension=extension,
        timesys=timesys,
        timeref=timeref,
        minmjd=minmjd,
        maxmjd=maxmjd,
        errors=errors,
    )

    return toas.to_TOA_list()


def get_fits_TOAs(
    eventname,
    mission,
    weights=None,
    extension=None,
    timesys=None,
    timeref=None,
    minmjd=-np.inf,
    maxmjd=np.inf,
    ephem=None,
    planets=False,
    include_bipm=False,
    include_gps=False,
    errors=_default_uncertainty["default"],
):
    """
    Read photon event times out of a FITS file as :class:`pint.toa.TOAs` object

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered TOAs. Different conditions may apply to different missions.

    The minmjd/maxmjd parameters can be used to avoid instantiation of TOAs
    we don't want, which can otherwise be very slow.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    mission : str
        Name of the mission (e.g. RXTE, XMM)
    weights : array or None
        The array has to be of the same size as the event list. Overwrites
        possible weight lists from mission-specific FITS files
    extension : str
        FITS extension to read
    timesys : str, default None
        Force this time system
    timeref : str, default None
        Forse this time reference
    minmjd : float, default "-infinity"
        minimum MJD timestamp to return
    maxmjd : float, default "infinity"
        maximum MJD timestamp to return
    ephem : str, optional
        The name of the solar system ephemeris to use; defaults to "DE421".
    planets : bool, optional
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.
    include_bipm : bool, optional
        Use TT(BIPM) instead of TT(TAI)
    include_gps : bool, optional
        Apply GPS to UTC clock corrections
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    pint.toa.TOAs
    """
    # Load photon times from event file
    hdulist = pyfits.open(eventname)
    if mission not in mission_config:
        log.warning("Mission not recognized. Using generic")
        mission = "generic"

    if (
        extension is not None
        and isinstance(extension, str)
        and hdulist[1].name not in extension.split(",")
    ):
        raise RuntimeError(
            f"First table in FITS file must be {extension}. Found {hdulist[1].name}"
        )
    if isinstance(extension, int) and extension != 1:
        raise ValueError(
            "At the moment, only data in the first FITS extension is supported"
        )

    if timesys is None:
        timesys = _get_timesys(hdulist[1])
    if timeref is None:
        timeref = _get_timeref(hdulist[1])
    log.info(f"TIMESYS: {timesys} TIMEREF: {timeref}")
    check_timesys(timesys)
    check_timeref(timeref)

    if not mission_config[mission]["allow_local"] and timesys != "TDB":
        log.error(f"Raw spacecraft TOAs not yet supported for {mission}")

    obs, scale = _default_obs_and_scale(mission, timesys, timeref)

    # Read time column from FITS file
    mjds = read_fits_event_mjds_tuples(hdulist[1])

    new_kwargs = _get_columns_from_fits(
        hdulist[1], mission_config[mission]["fits_columns"]
    )

    hdulist.close()

    if weights is not None:
        new_kwargs["weights"] = weights

    if not isinstance(errors, u.Quantity):
        errors = errors * u.microsecond

    # mask out times/columns outside of mjd range
    mjds_float = np.asarray([r[0] + r[1] for r in mjds])
    idx = (minmjd < mjds_float) & (mjds_float < maxmjd)
    mjds = mjds[idx]
    for key in new_kwargs.keys():
        new_kwargs[key] = new_kwargs[key][idx]

    location = EarthLocation(0, 0, 0) if timeref == "GEOCENTRIC" else None

    if len(mjds.shape) == 2:
        t = Time(
            val=mjds[:, 0],
            val2=mjds[:, 1],
            format="mjd",
            scale=scale,
            location=location,
        )
    else:
        t = Time(mjds, format="mjd", scale=scale, location=location)
    flags = [toa.FlagDict() for _ in range(len(mjds))]
    for i in range(len(mjds)):
        for key in new_kwargs:
            flags[i][key] = str(new_kwargs[key][i])

    return toa.get_TOAs_array(
        t,
        obs,
        include_gps=include_gps,
        include_bipm=include_bipm,
        planets=planets,
        ephem=ephem,
        flags=flags,
        errors=errors,
    )


def load_event_TOAs(
    eventname,
    mission,
    weights=None,
    minmjd=-np.inf,
    maxmjd=np.inf,
    errors=_default_uncertainty["default"],
):
    """
    Read photon event times out of a FITS file as PINT :class:`~pint.toa.TOA` objects.

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered TOAs. Different conditions may apply to different missions.

    The minmjd/maxmjd parameters can be used to avoid instantiation of TOAs
    we don't want, which can otherwise be very slow.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    mission : str
        Name of the mission (e.g. RXTE, XMM)
    weights : array or None
        The array has to be of the same size as the event list. Overwrites
        possible weight lists from mission-specific FITS files
    minmjd : float, default "-infinity"
        minimum MJD timestamp to return
    maxmjd : float, default "infinity"
        maximum MJD timestamp to return
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    toalist : list of :class:`~pint.toa.TOA` objects

    Note
    ----
    This list should be converted into a :class:`~pint.toa.TOAs` object with :func:`pint.toa.get_TOAs_list` for most operations

    See Also
    --------
    :func:`get_event_TOAs`
    """
    # Load photon times from event file

    try:
        extension = mission_config[mission]["fits_extension"]
    except ValueError:
        log.warning("Mission name (TELESCOP) not recognized, using generic!")
        extension = mission_config["generic"]["fits_extension"]
    return load_fits_TOAs(
        eventname,
        mission,
        weights=weights,
        extension=extension,
        minmjd=minmjd,
        maxmjd=maxmjd,
        errors=errors,
    )


def get_event_TOAs(
    eventname,
    mission,
    weights=None,
    minmjd=-np.inf,
    maxmjd=np.inf,
    ephem=None,
    planets=False,
    include_bipm=False,
    include_gps=False,
    errors=_default_uncertainty["default"],
):
    """
    Read photon event times out of a FITS file as a :class:`pint.toa.TOAs` object

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered TOAs. Different conditions may apply to different missions.

    The minmjd/maxmjd parameters can be used to avoid instantiation of TOAs
    we don't want, which can otherwise be very slow.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    mission : str
        Name of the mission (e.g. RXTE, XMM)
    weights : array or None
        The array has to be of the same size as the event list. Overwrites
        possible weight lists from mission-specific FITS files
    minmjd : float, default "-infinity"
        minimum MJD timestamp to return
    maxmjd : float, default "infinity"
        maximum MJD timestamp to return
    ephem : str, optional
        The name of the solar system ephemeris to use; defaults to "DE421".
    planets : bool, optional
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.
    include_bipm : bool, optional
        Use TT(BIPM) instead of TT(TAI)
    include_gps : bool, optional
        Apply GPS to UTC clock corrections
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    pint.toa.TOAs

    """
    # Load photon times from event file

    try:
        extension = mission_config[mission]["fits_extension"]
    except ValueError:
        log.warning("Mission name (TELESCOP) not recognized, using generic!")
        extension = mission_config["generic"]["fits_extension"]
    return get_fits_TOAs(
        eventname,
        mission,
        weights=weights,
        extension=extension,
        minmjd=minmjd,
        maxmjd=maxmjd,
        ephem=ephem,
        planets=planets,
        include_bipm=include_bipm,
        include_gps=include_gps,
        errors=errors,
    )


# generic docstring for these functions
_load_event_docstring = """
    Read photon event times out of a {} file as PINT :class:`~pint.toa.TOA` objects.

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered TOAs. Different conditions may apply to different missions.

    The minmjd/maxmjd parameters can be used to avoid instantiation of TOAs
    we don't want, which can otherwise be very slow.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    minmjd : float, default "-infinity"
        minimum MJD timestamp to return
    maxmjd : float, default "infinity"
        maximum MJD timestamp to return
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    toalist : list of :class:`~pint.toa.TOA` objects

    Note
    ----
    This list should be converted into a :class:`~pint.toa.TOAs` object with :func:`pint.toa.get_TOAs_list` for most operations

    See Also
    --------
    :func:`get_{}_TOAs`
    :func:`load_event_TOAs`
    """

load_RXTE_TOAs = partial(
    load_event_TOAs, mission="xte", errors=_default_uncertainty["RXTE"]
)
load_RXTE_TOAs.__doc__ = _load_event_docstring.format("RXTE", "RXTE")

load_NICER_TOAs = partial(
    load_event_TOAs, mission="nicer", errors=_default_uncertainty["NICER"]
)
load_NICER_TOAs.__doc__ = _load_event_docstring.format("NICER", "NICER")

load_IXPE_TOAs = partial(
    load_event_TOAs, mission="ixpe", errors=_default_uncertainty["IXPE"]
)
load_IXPE_TOAs.__doc__ = _load_event_docstring.format("IXPE", "IXPE")

load_XMM_TOAs = partial(
    load_event_TOAs, mission="xmm", errors=_default_uncertainty["XMM"]
)
load_XMM_TOAs.__doc__ = _load_event_docstring.format("XMM", "XMM")

load_NuSTAR_TOAs = partial(
    load_event_TOAs, mission="nustar", errors=_default_uncertainty["NuSTAR"]
)
load_NuSTAR_TOAs.__doc__ = _load_event_docstring.format("NuSTAR", "NuSTAR")

load_Swift_TOAs = partial(
    load_event_TOAs, mission="swift", errors=_default_uncertainty["Swift"]
)
load_Swift_TOAs.__doc__ = _load_event_docstring.format("Swift", "Swift")

# generic docstring for these functions
_get_event_docstring = """
    Read photon event times out of a {} file as a :class:`pint.toa.TOAs` object

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered TOAs. Different conditions may apply to different missions.

    The minmjd/maxmjd parameters can be used to avoid instantiation of TOAs
    we don't want, which can otherwise be very slow.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    minmjd : float, default "-infinity"
        minimum MJD timestamp to return
    maxmjd : float, default "infinity"
        maximum MJD timestamp to return
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds
    ephem : str, optional
        The name of the solar system ephemeris to use; defaults to "DE421".
    planets : bool, optional
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.

    Returns
    -------
    pint.toa.TOAs        

    See Also
    --------
    :func:`get_event_TOAs`        
    """

get_RXTE_TOAs = partial(
    get_event_TOAs, mission="xte", errors=_default_uncertainty["RXTE"]
)
get_RXTE_TOAs.__doc__ = _get_event_docstring.format("RXTE")

get_NICER_TOAs = partial(
    get_event_TOAs, mission="nicer", errors=_default_uncertainty["NICER"]
)
get_NICER_TOAs.__doc__ = _get_event_docstring.format("NICER")

get_IXPE_TOAs = partial(
    get_event_TOAs, mission="ixpe", errors=_default_uncertainty["IXPE"]
)
get_IXPE_TOAs.__doc__ = _get_event_docstring.format("IXPE")

get_XMM_TOAs = partial(
    get_event_TOAs, mission="xmm", errors=_default_uncertainty["XMM"]
)
get_XMM_TOAs.__doc__ = _get_event_docstring.format("XMM")

get_NuSTAR_TOAs = partial(
    get_event_TOAs, mission="nustar", errors=_default_uncertainty["NuSTAR"]
)
get_NuSTAR_TOAs.__doc__ = _get_event_docstring.format("NuSTAR")

get_Swift_TOAs = partial(
    get_event_TOAs, mission="swift", errors=_default_uncertainty["Swift"]
)
get_Swift_TOAs.__doc__ = _get_event_docstring.format("Swift")
