#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

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

def phaseogram(mjds, phases, weights=None, title=None, bins=100, rotate=0.0, size=5,
    alpha=0.25, width=6, maxphs=2.0, file=False):
    """
    Make a nice 2-panel phaseogram
    """
    years = (mjds - 51544.0) / 365.25 + 2000.0
    phss = phases + rotate
    phss[phss > 1.0] -= 1.0
    fig = plt.figure(figsize=(width, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    wgts = None if weights is None else np.concatenate((weights, weights))
    h, x, p = ax1.hist(np.concatenate((phss, phss+1.0)),
        int(maxphs*bins), range=[0,maxphs], weights=wgts,
        color='k', histtype='step', fill=False, lw=2)
    ax1.set_xlim([0.0, maxphs]) # show 1 or more pulses
    ax1.set_ylim([0.0, 1.1*h.max()])
    if weights is not None:
        ax1.set_ylabel("Weighted Counts")
    else:
        ax1.set_ylabel("Counts")
    if title is not None:
        ax1.set_title(title)
    if weights is None:
        ax2.scatter(phss, mjds, s=size, color='k', alpha=alpha)
        ax2.scatter(phss+1.0, mjds, s=size, color='k', alpha=alpha)
    else:
        colarray = np.array([[0.0,0.0,0.0,w] for w in weights])
        ax2.scatter(phss, mjds, s=size, color=colarray)
        ax2.scatter(phss+1.0, mjds, s=size, color=colarray)
    ax2.set_xlim([0.0, maxphs]) # show 1 or more pulses
    ax2.set_ylim([mjds.min(), mjds.max()])
    ax2.set_ylabel("MJD")
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_scientific(False)
    ax2r = ax2.twinx()
    ax2r.set_ylim([years.min(), years.max()])
    ax2r.set_ylabel("Year")
    ax2r.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2r.get_yaxis().get_major_formatter().set_scientific(False)
    ax2.set_xlabel("Pulse Phase")
    plt.tight_layout()
    if file:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

def load_Fermi_TOAs(ft1name, ft2name=None, weightcolumn=None, targetcoord=None,
                    logeref=4.1, logesig=0.5, minweight=0.0,
                    minmjd=0.0, maxmjd=np.inf):
    '''
    TOAlist = load_Fermi_TOAs(ft1name,ft2name=None)
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
    import pyfits
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

    # Collect TIMEZERO and MJDREF
    try:
        TIMEZERO = np.longdouble(ft1hdr['TIMEZERO'])
    except KeyError:
        TIMEZERO = np.longdouble(ft1hdr['TIMEZERI']) + np.longdouble(ft1hdr['TIMEZERF'])
    #print >>outfile, "# TIMEZERO = ",TIMEZERO
    log.info("TIMEZERO = {0}".format(TIMEZERO))
    try:
        MJDREF = np.longdouble(ft1hdr['MJDREF'])
    except KeyError:
        # Here I have to work around an issue where the MJDREFF key is stored
        # as a string in the header and uses the "1.234D-5" syntax for floats, which
        # is not supported by Python
        if isinstance(ft1hdr['MJDREFF'],basestring):
            MJDREF = np.longdouble(ft1hdr['MJDREFI']) + \
            np.longdouble(ft1hdr['MJDREFF'].replace('D','E'))
        else:
            MJDREF = np.longdouble(ft1hdr['MJDREFI']) + np.longdouble(ft1hdr['MJDREFF'])
    #print >>outfile, "# MJDREF = ",MJDREF
    log.info("MJDREF = {0}".format(MJDREF))
    mjds = np.array(ft1dat.field('TIME'),dtype=np.longdouble)/86400.0 + MJDREF + TIMEZERO
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
    # limit the TOAs to ones in selected MJD range
    idx = (mjds > minmjd) & (mjds < maxmjd)
    mjds = mjds[idx]
    energies = energies[idx]
    if weightcolumn is not None:
        weights = weights[idx]

    if timesys == 'TDB':
        log.info("Building barycentered TOAs")
        if weightcolumn is None:
            toalist=[toa.TOA(m,obs='Barycenter',scale='tdb',energy=e) for m,e in zip(mjds,energies)]
        else:
            toalist=[toa.TOA(m,obs='Barycenter',scale='tdb',energy=e,weight=w) for m,e,w in zip(mjds,energies,weights)]
    else:
        if timeref == 'LOCAL':
            log.info('LOCAL TOAs not implemented yet')
            if ft2name is None:
                log.error('FT2 file required to process raw Fermi times.')
            if weightcolumn is None:
                toalist=[toa.TOA(m, obs='Spacecraft', scale='tt',energy=e) for m,e in zip(mjds,energies)]
            else:
                toalist=[toa.TOA(m, obs='Spacecraft', scale='tt',energy=e,weight=w) for m,e,w in zip(mjds,energies,weights)]
        else:
            log.info("Building geocentered TOAs")
            if weightcolumn is None:
                toalist=[toa.TOA(m, obs='Geocenter', scale='tt',energy=e) for m,e in zip(mjds,energies)]
            else:
                toalist=[toa.TOA(m, obs='Geocenter', scale='tt',energy=e,weight=w) for m,e,w in zip(mjds,energies,weights)]

    return toalist

if __name__ == '__main__':
    ephem = 'DE421'
    planets = True
    parfile = 'PSRJ0030+0451_psrcat.par'
    #eventfile = 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_short.fits'
    eventfile = 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_BARY.fits'
    #weightcol = 'PSRJ0030+0451'
    weightcol = None # 'CALC'

    #eventfile = 'J0740+6620_P8_15.0deg_239557517_458611204_ft1weights_GEO_short.fits'
    #parfile = 'J0740+6620.par'
    #weightcol = 'PSRJ0740+6620'

    # Read event file and return list of TOA objects
    tl  = load_Fermi_TOAs(eventfile,weightcolumn=weightcol,targetcoord=SkyCoord('00:30:27.4303','+04:51:39.74',unit=(u.hourangle,u.degree),frame='icrs'))

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
    if weightcol is not None:
        weights = np.array([w['weight'] for w in ts.table['flags']])
        phaseogram(mjds,phases,weights)
    else:
        phaseogram(mjds,phases)
