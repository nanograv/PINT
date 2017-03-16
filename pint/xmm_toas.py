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


def load_XMM_TOAs(eventname):
    '''
    TOAlist = load_XMM_TOAs(eventname)
      Read photon event times out of a XMM event FITS file and return
      a list of PINT TOA objects.
      Currently only handles barycentered TOAs (produced by the SAS tool barycen)
      Eventually should be extended to handle raw events, but that requires
      understanding the orbit data in the XMM data products.

    '''
    import astropy.io.fits as pyfits
    # Load photon times from FT1 file
    hdulist = pyfits.open(eventname)

    # This code currently supports XMM science event data
    if hdulist[1].name not in ['EVENTS']:
        raise RuntimeError('For XMM data, first table in FITS file must be EVENTS. Found '+hdulist[1].name)
    event_hdr=hdulist[1].header
    event_dat=hdulist[1].data

    if not event_hdr['TELESCOP'].startswith('XMM'):
        log.error('XMM data should have TELESCOP == XMM, found '+event_hdr['TELESCOP'])

    # TIMESYS will be 'TT' for unmodified XMM events (or geocentered), and
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
        phas = event_dat.field('PI')
    except:
        phas = np.zeros(len(mjds))

    hdulist.close()

    if timesys == 'TDB':
        log.info("Building barycentered TOAs")
        toalist=[toa.TOA(m,obs='Barycenter',scale='tdb',pi=e) for m,e in zip(mjds,phas)]
    else:
        if timeref == 'LOCAL':
            log.error('XMM raw spacecraft TOAs not yet supported')
            raise NotImplementedError
        else:
            log.info("Assuming geocentered TOAs")
            toalist=[toa.TOA(m, obs='Geocenter', scale='tt',pi=e) for m,e in zip(mjds,phas)]

    return toalist
