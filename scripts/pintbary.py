#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
from __future__ import division, print_function
import numpy as np
import pint.toa as toa
import pint.models
from astropy.coordinates import SkyCoord
from astropy import log
from astropy.time import Time
import argparse
from pint import pulsar_mjd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PINT tool for command-line barycentering calculations.")

    parser.add_argument("time",help="MJD (UTC, by default)")
    parser.add_argument("--timescale",default="utc",help="Time scale for MJD argument ('utc', 'tt', 'tdb'), default=utc")
    parser.add_argument("--format",help="Format for time argument",default="mjd")
    parser.add_argument("--freq",type=float,default=np.inf)
    parser.add_argument("--obs",default="Geocenter")
    parser.add_argument("--parfile",help="par file to read model from",default=None)
    parser.add_argument("--ra",help="RA to use (if not read from par file)")
    parser.add_argument("--dec",help="Decl. to use (if not read from par file)")
    parser.add_argument("--dm",help="DM to use (if not read from par file)")
    parser.add_argument("--ephem",default="DE421",help="Ephemeris to use")

    args = parser.parse_args()

    if args.format in ("mjd","jd"):
        # These formats require conversion from string to longdouble first
        t = Time(np.longdouble(args.time),scale=args.timescale,format=args.format,
            precision=9)
    else:
        t = Time(args.time,scale=args.timescale,format=args.format, precision=9)
    print(t.iso, isinstance(t,Time))

    t = toa.TOA(t,freq=args.freq,obs=args.obs)

    if args.parfile is not None:
        m=pint.models.get_model(args.parfile)
    else:
        # Construct model by hand
        m=pint.models.StandardTimingModel()

    ts = toa.TOAs(toalist=[t])
    ts.compute_TDBs()
    ts.compute_posvels()
    tdbtimes = m.get_barycentric_toas(ts.table)

    print(tdbtimes[0])
    
