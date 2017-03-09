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

from astropy import log

def nicer_phaseogram(mjds, phases, weights=None, title=None, bins=64, rotate=0.0, size=5,
    alpha=0.25, width=6, maxphs=2.0, file=False):
    """
    Make a nice 2-panel phaseogram
    """
    import matplotlib.pyplot as plt
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
    SCATTER=False
    if SCATTER:
        if weights is None:
            ax2.scatter(phss, mjds, s=size, color='k', alpha=alpha)
            ax2.scatter(phss+1.0, mjds, s=size, color='k', alpha=alpha)
        else:
            colarray = np.array([[0.0,0.0,0.0,w] for w in weights])
            ax2.scatter(phss, mjds, s=size, color=colarray)
            ax2.scatter(phss+1.0, mjds, s=size, color=colarray)
    else:
        profile = np.zeros(bins,dtype=np.float_)
        ntoa = 64
        toadur = (mjds.max()-mjds.min())/ntoa
        mjdstarts = mjds.min() + toadur*np.arange(ntoa,dtype=np.float_)
        mjdstops = mjdstarts + toadur
        # Loop over blocks to process
        a = []
        for tstart,tstop in zip(mjdstarts,mjdstops):

            # Clear profile array
            profile = profile*0.0

            idx = (mjds>tstart)&(mjds<tstop)

            if weights is not None:
                for ph,ww in zip(phases[idx],weights[idx]):
                    bin = int(ph*bins)
                    profile[bin] += ww
            else:
                for ph in phases[idx]:
                    bin = int(ph*bins)
                    profile[bin] += 1

            for i in xrange(bins):
                a.append(profile[i])

        a = np.array(a)
        b = a.reshape(ntoa,bins)
        c = np.hstack([b,b])
        ax2.imshow(c, interpolation='nearest', origin='lower', cmap=plt.cm.binary,
             extent=(0,2.0,mjds.min(),mjds.max()),aspect='auto')

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
    # This code currently support NICER and RXTE science event data
    if hdulist[1].name not in ['EVENTS', 'XTE_SE']:
        raise RuntimeError('First table in FITS file must be EVENTS or XTE_SE. Found '+hdulis[1].name)
    event_hdr=hdulist[1].header
    event_dat=hdulist[1].data

    # TIMESYS will be 'TT' for unmodified NICER events (or geocentered), and
    #                 'TDB' for events barycentered with axBary
    # TIMEREF will be 'GEOCENTER' for geocentered events,
    #                 'SOLARSYSTEM' for barycentered,
    #             and 'LOCAL' for unmodified events

    timesys = event_hdr['TIMESYS']
    log.info("TIMESYS {0}".format(timesys))
    timeref = event_hdr['TIMEREF']
    log.info("TIMEREF {0}".format(timeref))

    # Collect TIMEZERO and MJDREF
    # IMPORTANT: TIMEZERO is in SECONDS (not days)!
    try:
        TIMEZERO = np.longdouble(event_hdr['TIMEZERO'])
    except KeyError:
        TIMEZERO = np.longdouble(event_hdr['TIMEZERI']) + np.longdouble(event_hdr['TIMEZERF'])
    log.info("TIMEZERO = {0}".format(TIMEZERO))
    try:
        MJDREF = np.longdouble(event_hdr['MJDREF'])
    except KeyError:
        # Here I have to work around an issue where the MJDREFF key is stored
        # as a string in the header and uses the "1.234D-5" syntax for floats, which
        # is not supported by Python
        if isinstance(event_hdr['MJDREFF'],six.string_types):
            MJDREF = np.longdouble(event_hdr['MJDREFI']) + \
            np.longdouble(event_hdr['MJDREFF'].replace('D','E'))
        else:
            MJDREF = np.longdouble(event_hdr['MJDREFI']) + np.longdouble(event_hdr['MJDREFF'])
    log.info("MJDREF = {0}".format(MJDREF))
    mjds = (np.array(event_dat.field('TIME'),dtype=np.longdouble)+ TIMEZERO)/86400.0 + MJDREF 
    
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
    nicer_phaseogram(mjds,phases)
