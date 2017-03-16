#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.extern import six
from pint.fits_utils import read_fits_event_mjds
from astropy import log


def load_RXTE_TOAs(eventname):
    '''
    TOAlist = load_RXTE_TOAs(eventname)
      Read photon event times out of a RXTE event FITS file and return
      a list of PINT TOA objects.
      Correctly handles raw RXTE files, or ones processed with axBary
      to have barycentered  TOAs.

    '''
    import astropy.io.fits as pyfits
    # Load photon times from FT1 file
    hdulist = pyfits.open(eventname)

    # This code currently supports RXTE science event data
    if hdulist[1].name not in ['XTE_SE']:
        raise RuntimeError('First table in FITS file must be XTE_SE. Found '+hdulist[1].name)
    event_hdr=hdulist[1].header
    event_dat=hdulist[1].data

    if not event_hdr['TELESCOP'].startswith('XTE'):
        log.error('RXTE data should have TELESCOP == RXTE, found '+event_hdr['TELESCOP'])
        raise(RuntimeError)

    # TIMESYS will be 'TT' for unmodified RXTE events (or geocentered), and
    #                 'TDB' for events barycentered with axBary
    # TIMEREF will be 'GEOCENTER' for geocentered events,
    #                 'SOLARSYSTEM' for barycentered,
    #             and 'LOCAL' for unmodified events

    timesys = event_hdr['TIMESYS']
    log.info("TIMESYS {0}".format(timesys))
    timeref = event_hdr['TIMEREF']
    log.info("TIMEREF {0}".format(timeref))

    # Read time column from FITS file
    mjds = read_fits_event_mjds(hdulist[1])

    try:
        phas = event_dat.field('PHA')
    except:
        phas = np.zeros(len(mjds))

    hdulist.close()

    if timesys == 'TDB':
        log.info("Building barycentered TOAs")
        toalist=[toa.TOA(m,obs='Barycenter',scale='tdb',energy=e) for m,e in zip(mjds,phas)]
    else:
        if timeref == 'LOCAL':
            log.info('Building spacecraft local TOAs, with MJDs in range {0} to {1}'.format(mjds.min(),mjds.max()))
            toalist=[toa.TOA(m, obs='RXTE', scale='tt',energy=e) for m,e in zip(mjds,phas)]
        else:
            log.info("Building geocentered TOAs")
            toalist=[toa.TOA(m, obs='Geocenter', scale='tt',energy=e) for m,e in zip(mjds,phas)]

    return toalist
