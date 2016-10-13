#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""PINT-based tool for making simulated TOAs

"""
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.fitter
import astropy.units as u
from astropy.time import Time, TimeDelta
import argparse

from astropy import log
log.setLevel('DEBUG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("parfile",help="par file to read model from")
    parser.add_argument("timfile",help="Output TOA file name")
    parser.add_argument("--startMJD",help="MJD of first fake TOA (default=56000.0)",
                    type=float, default=56000.0)
    parser.add_argument("--ntoa",help="Number of fake TOAs to generate",type=int,default=100)
    parser.add_argument("--duration",help="Span of TOAs to generate (days)",type=int,default=400)
    parser.add_argument("--obs",help="Observatory code (default: GBT)",default="GBT")
    parser.add_argument("--freq",help="Frequency for TOAs (MHz) (default: 1400)",
                    type=float, default=1400.0)
    parser.add_argument("--error",help="Random error to apply to each TOA (us, default=1.0)",
                    type=float, default=1.0)
    parser.add_argument("--plot",help="Plot residuals",action="store_true",default=False)
    parser.add_argument("--ephem",help="Ephemeris to use",default="DE421")
    parser.add_argument("--planets",help="Use planetary Shapiro delay",action="store_true",
                        default=False)
    args = parser.parse_args()

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    duration = args.duration*u.day
    start = Time(args.startMJD,scale='utc',format='mjd',precision=9)
    error = args.error*u.s
    freq = args.freq*u.MHz
    scale = 'utc'

    times = np.linspace(0,duration.to(u.day).value,args.ntoa)*u.day + start

    tl = [toa.TOA(t,error=error, obs=args.obs, freq=freq,
                 scale=scale) for t in times]
    del t
    ts = toa.TOAs(toalist=tl)

    # WARNING! I'm not sure how clock corrections should be handled here!
    # Do we apply them, or not?
    if not any([f.has_key('clkcorr') for f in ts.table['flags']]):
        log.info("Applying clock corrections.")
        ts.apply_clock_corrections()
    if 'tdb' not in ts.table.colnames:
        log.info("Getting IERS params and computing TDBs.")
        ts.compute_TDBs()
    if 'ssb_obs_pos' not in ts.table.colnames:
        log.info("Computing observatory positions and velocities.")
        ts.compute_posvels(args.ephem, args.planets)

    # This computation should be replaced with a call to d_phase_d_TOA() when that
    # function works to compute the instantaneous topocentric frequency
    F = m.F0.value*m.F0.units    
    rs = m.phase(ts.table).frac/F

    # Adjust the TOA times to put them where their residuals will be 0.0
    ts.adjust_TOAs(TimeDelta(-1.0*rs))
    rspost = m.phase(ts.table).frac/F

    # Do a second iteration to fix the poor assumption of F = F0 when
    # converting phase residuals to time residuals
    ts.adjust_TOAs(TimeDelta(-1.0*rspost))

     # Write TOAs to a file
    ts.write_TOA_file(args.timfile,name='fake',format='Tempo2')

    if args.plot:
        import matplotlib.pyplot as plt
        rspost2 = m.phase(ts.table).frac/F
        plt.plot(ts.get_mjds(),rspost2.to(u.us),'+')
        plt.plot(ts.get_mjds(),rspost.to(u.us),'x')
        plt.show()
