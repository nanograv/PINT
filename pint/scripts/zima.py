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

from astropy import log
log.setLevel('INFO')

def main(argv=None):
    import argparse
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
    args = parser.parse_args(argv)

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    duration = args.duration*u.day
    start = Time(args.startMJD,scale='utc',format='mjd',precision=9)
    error = args.error*u.microsecond
    freq = args.freq*u.MHz
    scale = 'utc'

    times = np.linspace(0,duration.to(u.day).value,args.ntoa)*u.day + start

    tl = [toa.TOA(t,error=error, obs=args.obs, freq=freq,
                 scale=scale) for t in times]

    ts = toa.TOAs(toalist=tl)

    # WARNING! I'm not sure how clock corrections should be handled here!
    # Do we apply them, or not?
    if not any(['clkcorr' in f for f in ts.table['flags']]):
        log.info("Applying clock corrections.")
        ts.apply_clock_corrections()
    if 'tdb' not in ts.table.colnames:
        log.info("Getting IERS params and computing TDBs.")
        ts.compute_TDBs()
    if 'ssb_obs_pos' not in ts.table.colnames:
        log.info("Computing observatory positions and velocities.")
        ts.compute_posvels(args.ephem, args.planets)

    F_local = m.d_phase_d_toa(ts)*u.Hz
    rs = m.phase(ts.table).frac/F_local

    # Adjust the TOA times to put them where their residuals will be 0.0
    ts.adjust_TOAs(TimeDelta(-1.0*rs))
    rspost = m.phase(ts.table).frac/F_local

    # Do a second iteration
    ts.adjust_TOAs(TimeDelta(-1.0*rspost))

     # Write TOAs to a file
    #ts.write_TOA_file(args.timfile,name='fake',format='Tempo2')
    ts.write_TOA_file(args.timfile,name='fake',format='Tempo2')

    if args.plot:
        # This should be a very boring plot with all residuals flat at 0.0!
        import matplotlib.pyplot as plt
        rspost2 = m.phase(ts.table).frac/F_local
        plt.errorbar(ts.get_mjds(),rspost2.to(u.us).value,yerr=ts.get_errors().to(u.us).value)
        newts = pint.toa.get_TOAs(args.timfile)
        rsnew = m.phase(newts.table).frac/F_local
        plt.errorbar(newts.get_mjds(),rsnew.to(u.us).value,yerr=newts.get_errors().to(u.us).value)
        #plt.plot(ts.get_mjds(),rspost.to(u.us),'x')
        plt.xlabel('MJD')
        plt.ylabel('Residual (us)')
        plt.show()
