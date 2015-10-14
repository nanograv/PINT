#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
import matplotlib.pyplot as plt
from pint.fermi_toas import phaseogram, load_Fermi_TOAs

from astropy import log

if __name__ == '__main__':
    ephem = 'DE421'
    planets = False

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

    #print(ts.table)

    # Read in initial model
    modelin = pint.models.get_model(parfile)

    # Remove the dispersion delay as it is unnecessary
    modelin.delay_funcs.remove(modelin.dispersion_delay)

    # Compute model phase for each TOA
    phss = modelin.phase(ts.table)[1]
    # ensure all postive
    phases = np.where(phss < 0.0, phss + 1.0, phss)
    mjds = ts.get_mjds(high_precision=False)
    weights = np.array([w['weight'] for w in ts.table['flags']])
    phaseogram(mjds,phases,weights)
