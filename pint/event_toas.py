"""Generic function to load TOAs from events files."""
from __future__ import division, print_function

import numpy as np
import pint.toa as toa
from astropy import log
import astropy.io.fits as pyfits
from .fits_utils import read_fits_event_mjds_tuples

# fits_extension can be a single name or a comma-separated list of allowed
# extension names.
# For weight we use the same conventions used for Fermi: None, a valid FITS
# extension name or CALC.
mission_config = \
    {"rxte": {"fits_extension": "XTE_SE", 'allow_local': True,
              "fits_columns": {"pha": "PHA"}},
     "nicer": {"fits_extension": "EVENTS", 'allow_local': True,
               "fits_columns": {"pha": "PHA"}},
     "xmm": {"fits_extension": "EVENTS", 'allow_local': False,
             "fits_columns": {"pi": "PI"}}}


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
    if timesys == 'TDB':
        log.info("Building barycentered TOAs")
        obs, scale = 'Barycenter', 'tdb'
    elif timeref == 'LOCAL':
        log.info(
            'Building spacecraft local TOAs')
        obs, scale = mission, 'tt'
    else:
        log.info("Building geocentered TOAs")
        obs, scale = 'Geocenter', 'tt'

    return obs, scale


def load_event_TOAs(eventname, mission, weights=None):
    '''
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
    '''
    # Load photon times from event file
    hdulist = pyfits.open(eventname)

    extension = mission_config[mission]["fits_extension"]
    allow_local = mission_config[mission]['allow_local']

    if hdulist[1].name not in extension.split(','):
        raise RuntimeError('First table in FITS file' +
                           'must be {}. Found {}'.format(extension,
                                                         hdulist[1].name))
    event_hdr=hdulist[1].header
    event_dat=hdulist[1].data

    timesys = event_hdr['TIMESYS']
    log.info("TIMESYS {0}".format(timesys))
    if timesys not in ['TDB', 'TT']:
        raise ValueError('Timesys has to be TDB or TT')

    timeref = event_hdr['TIMEREF']
    log.info("TIMEREF {0}".format(timeref))
    if timeref not in ['GEOCENTER', 'SOLARSYSTEM', 'LOCAL']:
        raise ValueError('Timeref is invalid')

    if allow_local is False and timesys != 'TDB':
        log.error('Raw spacecraft TOAs not yet supported for ' + mission)
    # Read time column from FITS file
    mjds = read_fits_event_mjds_tuples(hdulist[1])

    new_kwargs = {}
    # Parse and retrieve default values from the FITS columns listed in config
    cols = mission_config[mission]["fits_columns"]
    for col in cols.keys():
        try:
            val = event_dat.field(cols[col])
        except ValueError:
            val = np.zeros(len(mjds))
        new_kwargs[col] = val

    if weights is not None:
        new_kwargs["weights"] = weights

    hdulist.close()

    obs, scale = _default_obs_and_scale(mission, timesys, timeref)

    toalist = [None] * len(mjds)
    for i in range(len(mjds)):
        # Create TOA list
        kw = {}
        for key in new_kwargs.keys():
            kw[key] = new_kwargs[key][i]
        toalist[i] = toa.TOA(mjds[i], obs=obs, scale=scale, **kw)

    return toalist


def load_RXTE_TOAs(eventname):
    return load_event_TOAs(eventname, 'rxte')


def load_NICER_TOAs(eventname):
    return load_event_TOAs(eventname, 'nicer')


def load_XMM_TOAs(eventname):
    return load_event_TOAs(eventname, 'xmm')
