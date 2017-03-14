#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.extern import six
from pint.fits_utils import read_fits_event_mjds, read_fits_event_mjds_tuples
from pint.observatory import get_observatory

from astropy import log

def calc_lat_weights(energies, angseps, logeref=4.1, logesig=0.5):
    """
    This function computes photon weights based on the energy-dependent
    PSF, as defined in Philippe Bruel's SearchPulsation code.
    It was built by David Smith, based on some code from Lucas Guillemot.
    This computation uses only the PSF as a function of energy, not a full
    spectral model of the region, so is less exact than gtsrcprob.

    The input values are:
    energies : Array of photon energies in MeV
    angseps : Angular separations between photon direction and target
              This should be astropy Angle array, such as returned from
              SkyCoord_photons.separation(SkyCoord_target)
    logeref : Parameter from SearchPulsation optimization
    logesig : Parameter from SearchPulsation optimization

    Returns a numpy array of weights (probabilities that the photons came
    from the target, based on the PSF).

    """
    # A few parameters that define the PSF shape
    psfpar0 =  5.445
    psfpar1 =  0.848
    psfpar2 =  0.084
    norm = 1.
    gam = 2.
    scalepsf = 3.

    logE = np.log10(energies)

    sigma = np.sqrt(psfpar0*psfpar0*np.power(100./energies, 2.*psfpar1) + psfpar2*psfpar2)/scalepsf

    fgeom = norm*np.power(1+angseps.degree*angseps.degree/2./gam/sigma/sigma, -gam)

    return fgeom * np.exp(-np.power((logE-logeref)/np.sqrt(2.)/logesig,2.))

def load_Fermi_TOAs(ft1name,weightcolumn=None,targetcoord=None,logeref=4.1, logesig=0.5,minweight=0.0):
    '''
    TOAlist = load_Fermi_TOAs(ft1name)
      Read photon event times out of a Fermi FT1 file and return
      a list of PINT TOA objects.
      Correctly handles raw FT1 files, or ones processed with gtbary
      to have barycentered or geocentered TOAs.

      weightcolumn specifies the FITS column name to read the photon weights
      from.  The special value 'CALC' causes the weights to be computed empirically
      as in Philippe Bruel's SearchPulsation code.
      logeref and logesig are parameters for the weight computation and are only
      used when weightcolumn='CALC'.

      When weights are loaded, or computed, events are filtered by weight >= minweight
    '''
    import astropy.io.fits as pyfits
    # Load photon times from FT1 file
    hdulist = pyfits.open(ft1name)
    ft1hdr=hdulist[1].header
    ft1dat=hdulist[1].data

    # TIMESYS will be 'TT' for unmodified Fermi LAT events (or geocentered), and
    #                 'TDB' for events barycentered with gtbary
    # TIMEREF will be 'GEOCENTER' for geocentered events,
    #                 'SOLARSYSTEM' for barycentered,
    #             and 'LOCAL' for unmodified events

    timesys = ft1hdr['TIMESYS']
    log.info("TIMESYS {0}".format(timesys))
    timeref = ft1hdr['TIMEREF']
    log.info("TIMEREF {0}".format(timeref))

    # Read time column from FITS file
    mjds = read_fits_event_mjds_tuples(hdulist[1])

    energies = ft1dat.field('ENERGY')*u.MeV
    if weightcolumn is not None:
        if weightcolumn == 'CALC':
            photoncoords = SkyCoord(ft1dat.field('RA')*u.degree,ft1dat.field('DEC')*u.degree,frame='icrs')
            weights = calc_lat_weights(ft1dat.field('ENERGY'), photoncoords.separation(targetcoord), logeref=4.1, logesig=0.5)
        else:
            weights = ft1dat.field(weightcolumn)
        if minweight > 0.0:
            idx = np.where(weights>minweight)[0]
            mjds = mjds[idx]
            energies = energies[idx]
            weights = weights[idx]

    if timesys == 'TDB':
        log.info("Building barycentered TOAs")
        if weightcolumn is None:
            toalist=[toa.TOA(m,obs='Barycenter',scale='tdb',energy=e) for m,e in zip(mjds,energies)]
        else:
            toalist=[toa.TOA(m,obs='Barycenter',scale='tdb',energy=e,weight=w) for m,e,w in zip(mjds,energies,weights)]
    else:
        if timeref == 'LOCAL':
            log.info('Building spacecraft local TOAs, with MJDs in range {0} to {1}'.format(mjds[0],mjds[-1]))
            assert timesys == 'TT'
            fermiobs = get_observatory('Fermi')

            try:
                if weightcolumn is None:
                    toalist=[toa.TOA(m, obs='Fermi', scale='tt',energy=e)
                            for m,e in zip(mjds,energies)]
                else:
                    toalist=[toa.TOA(m, obs='Fermi', scale='tt',energy=e,weight=w)
                            for m,e,w in zip(mjds,energies,weights)]
            except KeyError:
                log.error('Error processing Fermi TOAs. You may have forgotten to specify an FT2 file with --ft2')
                raise
        else:
            log.info("Building geocentered TOAs")
            if weightcolumn is None:
                toalist=[toa.TOA(m, obs='Geocenter', scale='tt',energy=e) for m,e in zip(mjds,energies)]
            else:
                toalist=[toa.TOA(m, obs='Geocenter', scale='tt',energy=e,weight=w) for m,e,w in zip(mjds,energies,weights)]

    return toalist
