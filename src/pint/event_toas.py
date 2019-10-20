"""Generic function to load TOAs from events files."""
from __future__ import absolute_import, division, print_function

import astropy.io.fits as pyfits
import numpy as np
from astropy import log

import pint.toa as toa
from pint.fits_utils import read_fits_event_mjds_tuples

__all__ = [
    "load_fits_TOAs",
    "load_event_TOAs",
    "load_NuSTAR_TOAs",
    "load_NICER_TOAs",
    "load_RXTE_TOAs",
    "load_XMM_TOAs",
]

# fits_extension can be a single name or a comma-separated list of allowed
# extension names.
# For weight we use the same conventions used for Fermi: None, a valid FITS
# extension name or CALC.
mission_config = {
    "rxte": {
        "fits_extension": "XTE_SE",
        "allow_local": True,
        "fits_columns": {"pha": "PHA"},
    },
    "nicer": {
        "fits_extension": "EVENTS",
        "allow_local": True,
        "fits_columns": {"pha": "PHA"},
    },
    "nustar": {
        "fits_extension": "EVENTS",
        "allow_local": True,
        "fits_columns": {"pi": "PI"},
    },
    "xmm": {
        "fits_extension": "EVENTS",
        "allow_local": False,
        "fits_columns": {"pi": "PI"},
    },
}


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
    eventname, mission, weights=None, extension=None, timesys=None, timeref=None
):
    """
    Read photon event times out of a FITS file as PINT TOA objects.

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered  TOAs. Different conditions may apply to different missions.

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

    Returns
    -------
    toalist : list of TOA objects
    """
    # Load photon times from event file
    hdulist = pyfits.open(eventname)

    if extension is not None and hdulist[1].name not in extension.split(","):
        raise RuntimeError(
            "First table in FITS file"
            + "must be {}. Found {}".format(extension, hdulist[1].name)
        )

    if timesys is None:
        timesys = _get_timesys(hdulist[1])
    if timeref is None:
        timeref = _get_timeref(hdulist[1])
    check_timesys(timesys)
    check_timeref(timeref)

    if not mission_config[mission]["allow_local"] and timesys != "TDB":
        log.error("Raw spacecraft TOAs not yet supported for " + mission)

    obs, scale = _default_obs_and_scale(mission, timesys, timeref)

    # Read time column from FITS file
    mjds = read_fits_event_mjds_tuples(hdulist[1])

    new_kwargs = _get_columns_from_fits(
        hdulist[1], mission_config[mission]["fits_columns"]
    )

    hdulist.close()

    if weights is not None:
        new_kwargs["weights"] = weights

    toalist = [None] * len(mjds)
    kw = {}
    for i in range(len(mjds)):
        # Create TOA list
        for key in new_kwargs.keys():
            kw[key] = new_kwargs[key][i]
        toalist[i] = toa.TOA(mjds[i], obs=obs, scale=scale, **kw)

    return toalist


def load_event_TOAs(eventname, mission, weights=None):
    """
    Read photon event times out of a FITS file as PINT TOA objects.

    Correctly handles raw event files, or ones processed with axBary to have
    barycentered  TOAs. Different conditions may apply to different missions.

    Parameters
    ----------
    eventname : str
        File name of the FITS event list
    mission : str
        Name of the mission (e.g. RXTE, XMM)
    weights : array or None
        The array has to be of the same size as the event list. Overwrites
        possible weight lists from mission-specific FITS files

    Returns
    -------
    toalist : list of TOA objects
    """
    # Load photon times from event file

    extension = mission_config[mission]["fits_extension"]
    return load_fits_TOAs(eventname, mission, weights=weights, extension=extension)


def load_RXTE_TOAs(eventname):
    return load_event_TOAs(eventname, "rxte")


def load_NICER_TOAs(eventname):
    return load_event_TOAs(eventname, "nicer")


def load_XMM_TOAs(eventname):
    return load_event_TOAs(eventname, "xmm")


def load_NuSTAR_TOAs(eventname):
    return load_event_TOAs(eventname, "nustar")
