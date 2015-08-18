#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
import matplotlib.pyplot as plt

from astropy import log

def phaseogram(mjds, phases, weights=None, bins=100, rotate=0.0, size=5,
    alpha=0.25, file=False):
    """
    Make a nice 2-panel phaseogram
    """
    years = (mjds - 51544.0) / 365.25 + 2000.0
    phss = phases + rotate
    phss[phss > 1.0] -= 1.0
    fig = plt.figure(figsize=(6,8))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    h, x, p = ax1.hist(np.concatenate((phss, phss+1.0)),
        2*bins, range=[0,2], weights=np.concatenate((weights, weights)),
        color='k', histtype='step', fill=False, lw=2)
    ax1.set_xlim([0.0, 2.0]) # show 2 pulses
    ax1.set_ylim([0.0, 1.1*h.max()])
    ax1.set_ylabel("Counts")
    #ax1.set_title(self.model.PSR.value)
    if weights is None:
        ax2.scatter(phss, mjds, s=size, color='k', alpha=alpha)
        ax2.scatter(phss+1.0, mjds, s=size, color='k', alpha=alpha)
    else:
        colarray = np.array([[0.0,0.0,0.0,w] for w in weights])
        ax2.scatter(phss, mjds, s=size, color=colarray)
        ax2.scatter(phss+1.0, mjds, s=size, color=colarray)
    ax2.set_xlim([0.0, 2.0]) # show 2 pulses
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

def load_Fermi_TOAs(ft1name,ft2name=None,weightcolumn=None):
    '''
    TOAlist = load_Fermi_TOAs(ft1name,ft2name=None)
      Read photon event times out of a Fermi FT1 file and return
      a list of PINT TOA objects.
      Correctly handles raw FT1 files, or ones processed with gtbary
      to have barycentered or geocentered TOAs.
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
        TIMEZERO = np.float128(ft1hdr['TIMEZERO'])
    except KeyError:
        TIMEZERO = np.float128(ft1hdr['TIMEZERI']) + np.float128(ft1hdr['TIMEZERF'])
    #print >>outfile, "# TIMEZERO = ",TIMEZERO
    log.info("TIMEZERO = {0}".format(TIMEZERO))
    try:
        MJDREF = np.float128(ft1hdr['MJDREF'])
    except KeyError:
        # Here I have to work around an issue where the MJDREFF key is stored
        # as a string in the header and uses the "1.234D-5" syntax for floats, which
        # is not supported by Python
        if isinstance(ft1hdr['MJDREFF'],basestring):
            MJDREF = np.float128(ft1hdr['MJDREFI']) + \
            np.float128(ft1hdr['MJDREFF'].replace('D','E'))
        else:
            MJDREF = np.float128(ft1hdr['MJDREFI']) + np.float128(ft1hdr['MJDREFF'])
    #print >>outfile, "# MJDREF = ",MJDREF
    log.info("MJDREF = {0}".format(MJDREF))
    mjds = np.array(ft1dat.field('TIME'),dtype=np.float128)/86400.0 + MJDREF + TIMEZERO
    if weightcolumn is not None:
        weights = ft1dat.field(weightcolumn)
    energies = ft1dat.field('ENERGY')*u.MeV

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
            toalist = []
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
    eventfile = 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_BARY_short.fits'
    weightcol = 'PSRJ0030+0451'

    #eventfile = 'J0740+6620_P8_15.0deg_239557517_458611204_ft1weights_GEO_short.fits'
    #parfile = 'J0740+6620.par'
    #weightcol = 'PSRJ0740+6620'

    # Read event file and return list of TOA objects
    tl  = load_Fermi_TOAs(eventfile,weightcolumn=weightcol)

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
    mjds = ts.get_mjds(high_precision=False)
    weights = np.array([w['weight'] for w in ts.table['flags']])
    phaseogram(mjds,phases,weights)
