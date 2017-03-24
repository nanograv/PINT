"""Generic function to load TOAs from events files."""
from __future__ import division, print_function

import numpy as np
import pint.toa as toa
from astropy import log
import astropy.io.fits as pyfits
from .fits_utils import read_fits_event_mjds_tuples

# fits_extension can be a single name or a comma-separated list of allowed
# extension names
mission_config = \
    {"rxte": {"fits_extension": "XTE_SE", 'allow_local': True},
     "nicer": {"fits_extension": "EVENTS", 'allow_local': True},
     "xmm": {"fits_extension": "EVENTS", 'allow_local': False}}


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

    try:
        phas = event_dat.field('PHA')
    except:
        phas = np.zeros(len(mjds))
    try:
        pis = event_dat.field('PI')
    except:
        pis = np.zeros(len(mjds), dtype=np.long)

    hdulist.close()

    obs, scale = _default_obs_and_scale(mission, timesys, timeref)

    # Create TOA list
    if weights is None:
        toalist = [toa.TOA(m, obs=obs, scale=scale, energy=e, pi=pis) for
                   m, e in zip(mjds, phas)]
    else:
        toalist = [toa.TOA(m, obs=obs, scale=scale, energy=e, pi=pis,
                           weight=w) for
                   m, e, w in zip(mjds, phas, weights)]

    return toalist


def load_RXTE_TOAs(eventname):
    return load_event_TOAs(eventname, 'rxte')


def load_NICER_TOAs(eventname):
    return load_event_TOAs(eventname, 'nicer')


def load_XMM_TOAs(eventname):
    return load_event_TOAs(eventname, 'xmm')
