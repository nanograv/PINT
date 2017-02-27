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
import argparse
from astropy.time import Time
from pint.eventstats import hmw, hm, h2sig
from astropy.coordinates import SkyCoord

from astropy import log

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Use PINT to compute H-test and plot Phaseogram from a Fermi FT1 event file.")
    parser.add_argument("eventfile",help="Fermi event FITS file name.  Should be GEOCENTERED.")
    parser.add_argument("parfile",help="par file to construct model from")
    parser.add_argument("weightcol",help="Column name for event weights (or 'CALC' to compute them)")
    parser.add_argument("--maxMJD",help="Maximum MJD to include in analysis", default=None)
    parser.add_argument("--outfile",help="Output figure file name (default=None)", default=None)
    parser.add_argument("--planets",help="Use planetary Shapiro delay in calculations (default=False)", default=False, action="store_true")
    parser.add_argument("--ephem",help="Planetary ephemeris to use (default=DE421)", default="DE421")
    args = parser.parse_args()


    # Read in model
    modelin = pint.models.get_model(args.parfile)
    if 'ELONG' in modelin.params:
        tc = SkyCoord(modelin.ELONG.quantity,modelin.ELAT.quantity,
            frame='barycentrictrueecliptic')
    else:
        tc = SkyCoord(modelin.RAJ.quantity,modelin.DECJ.quantity,frame='icrs')

    # Read event file and return list of TOA objects
    tl  = load_Fermi_TOAs(args.eventfile, weightcolumn=args.weightcol,
                          targetcoord=tc)

    # Discard events outside of MJD range    
    if args.maxMJD is not None:
        tlnew = []
        print("pre len : ",len(tl))
        maxT = Time(float(args.maxMJD),format='mjd')
        print("maxT : ",maxT)
        for tt in tl:
            if tt.mjd < maxT:
                tlnew.append(tt)
        tl=tlnew
        print("post len : ",len(tlnew))

    # Now convert to TOAs object and compute TDBs and posvels
    ts = toa.TOAs(toalist=tl)
    ts.filename = args.eventfile
    ts.compute_TDBs()
    ts.compute_posvels(ephem=args.ephem,planets=args.planets)

    print(ts.get_summary())
    mjds = ts.get_mjds()
    print(mjds.min(),mjds.max())

    # Compute model phase for each TOA
    phss = modelin.phase(ts.table)[1]
    # ensure all postive
    phases = np.where(phss < 0.0, phss + 1.0, phss)
    mjds = ts.get_mjds()
    weights = np.array([w['weight'] for w in ts.table['flags']])
    h = float(hmw(phases,weights))
    print("Htest : {0:.2f} ({1:.2f} sigma)".format(h,h2sig(h)))
    phaseogram(mjds,phases,weights,bins=100,file = args.outfile)
