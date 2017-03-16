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


def load_NICER_TOAs(eventname):
    '''
    TOAlist = load_NICER_TOAs(eventname)
      Read photon event times out of a NICER event FITS file and return
      a list of PINT TOA objects.
      Correctly handles raw NICER files, or ones processed with axBary
      to have barycentered  TOAs.

    '''
    import astropy.io.fits as pyfits
    # Load photon times from FT1 file
    hdulist = pyfits.open(eventname)

    # This code currently supports NICER science event data
    if hdulist[1].name not in ['EVENTS']:
        raise RuntimeError('For NICER data, first table in FITS file must be EVENTS. Found '+hdulist[1].name)
    event_hdr=hdulist[1].header
    event_dat=hdulist[1].data

    if not event_hdr['TELESCOP'].startswith('NICER'):
        log.error('NICER data should have TELESCOP == NICER, found '+event_hdr['TELESCOP'])

    # TIMESYS will be 'TT' for unmodified NICER events (or geocentered), and
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
            toalist=[toa.TOA(m, obs='NICER', scale='tt',energy=e) for m,e in zip(mjds,phas)]
        else:
            log.info("Building geocentered TOAs")
            toalist=[toa.TOA(m, obs='Geocenter', scale='tt',energy=e) for m,e in zip(mjds,phas)]

    return toalist

if __name__ == '__main__':
    ephem = 'DE421'
    planets = True
    parfile = 'PSRJ0030+0451_psrcat.par'
    eventfile = 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_BARY.fits'

    # Read event file and return list of TOA objects
    tl  = load_NICER_TOAs(eventfile)

    # Now convert to TOAs object and compute TDBs and posvels
    ts = toa.TOAs(toalist=tl)
    ts.filename = eventfile
    ts.compute_TDBs()
    ts.compute_posvels(ephem=ephem,planets=planets)

    print(ts.get_summary())

    print(ts.table)

    # Read in initial model
    modelin = pint.models.get_model(parfile)

    # Remove the dispersion delay as it is unnecessary
    modelin.delay_funcs.remove(modelin.dispersion_delay)

    # Hack to remove solar system delay terms if file was barycentered
    if 'Barycenter' in ts.observatories:
        modelin.delay_funcs.remove(modelin.solar_system_shapiro_delay)
        modelin.delay_funcs.remove(modelin.solar_system_geometric_delay)
    # Compute model phase for each TOA
    phss = modelin.phase(ts.table)[1]
    # ensure all postive
    phases = np.where(phss < 0.0, phss + 1.0, phss)
    mjds = ts.get_mjds()
    from pint.plot_utils import phaseogram_binned
    phaseogram_binned(mjds,phases)
