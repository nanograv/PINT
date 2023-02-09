#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
import argparse
import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

import pint.models
import pint.toa as toa
from pint.pulsar_mjd import Time

log = logging.getLogger(__name__)

__all__ = ["main"]

# log.setLevel("INFO")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool for command-line barycentering calculations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("time", help="MJD (UTC, by default)")
    parser.add_argument(
        "--timescale",
        default="utc",
        help="Time scale for MJD argument ('utc', 'tt', 'tdb'), default=utc",
    )
    parser.add_argument(
        "--format",
        help=(
            "Format for time argument ('mjd' or any astropy.Time format "
            "(e.g. 'isot'), see <http://docs.astropy.org/en/stable/time/#time-format>)"
        ),
        default="mjd",
    )
    parser.add_argument(
        "--freq", type=float, default=np.inf, help="Frequency to use, MHz"
    )
    parser.add_argument(
        "--obs", default="Geocenter", help="Observatory code (default = Geocenter)"
    )
    parser.add_argument("--parfile", help="par file to read model from", default=None)
    parser.add_argument(
        "--ra", help="RA to use (e.g. '12h22m33.2s' if not read from par file)"
    )
    parser.add_argument(
        "--dec", help="Decl. to use (e.g. '19d21m44.2s' if not read from par file)"
    )
    parser.add_argument(
        "--dm", help="DM to use (if not read from par file)", type=float, default=0.0
    )
    parser.add_argument("--ephem", default="DE421", help="Ephemeris to use")
    parser.add_argument(
        "--use_gps",
        default=False,
        action="store_true",
        help="Apply GPS to UTC clock corrections",
    )
    parser.add_argument(
        "--use_bipm",
        default=False,
        action="store_true",
        help="Use TT(BIPM) instead of TT(TAI)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=pint.logging.levels,
        default=pint.logging.script_level,
        help="Logging level",
        dest="loglevel",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", default=0, action="count", help="Decrease output verbosity"
    )

    args = parser.parse_args(argv)
    pint.logging.setup(
        level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet)
    )

    if args.format in ("mjd", "jd", "unix"):
        # These formats require conversion from string to long double first
        fmt = args.format
        # Never allow format == 'mjd' because it fails when scale is 'utc'
        # Change 'mjd' to 'pulsar_mjd' to deal with this.
        if fmt == "mjd":
            fmt = "pulsar_mjd"
        t = Time(
            np.longdouble(args.time), scale=args.timescale, format=fmt, precision=9
        )
        # print(t)
    else:
        t = Time(args.time, scale=args.timescale, format=args.format, precision=9)
    log.debug(t.iso)

    ts = toa.get_TOAs_array(
        t,
        freqs=args.freq,
        obs=args.obs,
        ephem=args.ephem,
        include_bipm=args.use_bipm,
        include_gps=args.use_gps,
        planets=False,
    )

    if args.parfile is not None:
        m = pint.models.get_model(args.parfile)
    else:
        # Construct model by hand
        m = pint.models.StandardTimingModel
        # Should check if 12:13:14.2 syntax is used and support that as well!
        m.RAJ.quantity = Angle(args.ra)
        m.DECJ.quantity = Angle(args.dec)
        m.DM.quantity = args.dm * u.parsec / u.cm**3

    tdbtimes = m.get_barycentric_toas(ts)

    print("{0:.16f}".format(tdbtimes[0].value))
    return
