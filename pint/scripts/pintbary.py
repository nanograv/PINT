#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
from __future__ import division, print_function
import sys
import numpy as np
import pint.toa as toa
import pint.models
from astropy.coordinates import SkyCoord, Angle
from astropy import log
from astropy.time import Time
from astropy import log
import astropy.units as u
import argparse
from pint import pulsar_mjd

log.setLevel('INFO')

def main(argv=None):
    parser = argparse.ArgumentParser(description="PINT tool for command-line barycentering calculations.")

    parser.add_argument("time",help="MJD (UTC, by default)")
    parser.add_argument("--timescale",default="utc",
        help="Time scale for MJD argument ('utc', 'tt', 'tdb'), default=utc")
    parser.add_argument("--format",
        help="Format for time argument ('mjd' or any astropy.Time format (e.g. 'isot'), see <http://docs.astropy.org/en/stable/time/#time-format>)",
        default="mjd")
    parser.add_argument("--freq",type=float,default=np.inf,
        help="Frequency to use, MHz")
    parser.add_argument("--obs",default="Geocenter", 
        help="Observatory code (default = Geocenter)")
    parser.add_argument("--parfile",help="par file to read model from",default=None)
    parser.add_argument("--ra",
        help="RA to use (e.g. '12h22m33.2s' if not read from par file)")
    parser.add_argument("--dec",
        help="Decl. to use (e.g. '19d21m44.2s' if not read from par file)")
    parser.add_argument("--dm",
        help="DM to use (if not read from par file)",type=float,default=0.0)
    parser.add_argument("--ephem",default="DE421",help="Ephemeris to use")

    args = parser.parse_args(argv)

    if args.format in ("mjd","jd"):
        # These formats require conversion from string to longdouble first
        fmt = args.format
        # Never allow format == 'mjd' because it fails when scale is 'utc'
        # Change 'mjd' to 'pulsar_mjd' to deal with this.
        if fmt == "mjd":
            fmt = "pulsar_mjd"
        t = Time(np.longdouble(args.time),scale=args.timescale,format=fmt,
            precision=9)
    else:
        t = Time(args.time,scale=args.timescale,format=args.format, precision=9)
    log.debug(t.iso)

    t = toa.TOA(t,freq=args.freq,obs=args.obs)
    # Build TOAs and compute TDBs and positions from ephemeris    
    ts = toa.TOAs(toalist=[t])
    ts.compute_TDBs()
    ts.compute_posvels(ephem=args.ephem)

    if args.parfile is not None:
        m=pint.models.get_model(args.parfile)
    else:
        # Construct model by hand
        m=pint.models.StandardTimingModel()
        # Should check if 12:13:14.2 syntax is used and support that as well!
        m.RAJ.quantity = Angle(args.ra)
        m.DECJ.quantity = Angle(args.dec)
        m.DM.quantity = args.dm*u.parsec/u.cm**3
        
    tdbtimes = m.get_barycentric_toas(ts.table)

    print("{0:.14f}".format(tdbtimes[0].value))
    return
