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
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
import argparse

from astropy import log
log.setLevel('DEBUG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("parfile",help="par file to read model from")
    parser.add_argument("timfile",help="Output TOA file name")
    parser.add_argument("--plot",help="Plot residuals",action="store_true",default=False)
    args = parser.parse_args()

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    ntoa = 100
    duration = 400*u.day
    start = Time(56000.0,scale='utc',format='mjd',precision=9)
    error = 1.0e-6*u.s
    freq = 1400.0*u.MHz
    scale = 'utc'
    obs = 'GBT'
    ephem = 'DE421'
    planets = False

    times = np.linspace(0,duration.to(u.day).value,ntoa)*u.day + start
    
    tl = [toa.TOA(t,error=error, obs=obs, freq=freq,
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
        ts.compute_posvels(ephem, planets)

                 
    F = m.F0.num_value*m.F0.num_unit    
    rs = m.phase(ts.table).frac/F
    #plt.plot(ts.get_mjds(),rs.to(u.us),'o')
    
    # Adjust the TOA times to put them where their residuals will be 0.0
    ts.adjust_TOAs(TimeDelta(-1.0*rs))

    rspost = m.phase(ts.table).frac/F
    plt.plot(ts.get_mjds(),rspost.to(u.us),'x')

    # Adjust the TOA times to put them where their residuals will be 0.0
    ts.adjust_TOAs(TimeDelta(-1.0*rspost))

    # Do a second iteration to fix the poor assumption of F = F0 when
    # converting phase residuals to time residuals
    rspost2 = m.phase(ts.table).frac/F
    plt.plot(ts.get_mjds(),rspost2.to(u.us),'+')

     # Write TOAs to a file
    ts.write_TOA_file(args.timfile,name='fake',format='Tempo2')
   
    plt.show()
    
    
