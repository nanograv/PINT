#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""PINT-based tool for making simulated TOAs."""
import logging

import astropy.units as u
import numpy as np
from astropy.time import TimeDelta

import pint.fitter
import pint.models
import pint.toa as toa
import pint.simulation
import pint.residuals

log = logging.getLogger(__name__)

__all__ = ["main"]

# log.setLevel("INFO")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="Output TOA file name")
    parser.add_argument(
        "--inputtim",
        help="Input tim file for fake TOA sampling",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--startMJD",
        help="MJD of first fake TOA (default=56000.0)",
        type=float,
        default=56000.0,
    )
    parser.add_argument(
        "--ntoa", help="Number of fake TOAs to generate", type=int, default=100
    )
    parser.add_argument(
        "--duration", help="Span of TOAs to generate (days)", type=float, default=400.0
    )
    parser.add_argument("--obs", help="Observatory code (default: GBT)", default="GBT")
    parser.add_argument(
        "--freq",
        help="Frequency for TOAs (MHz) (default: 1400)",
        nargs="+",
        type=float,
        default=1400.0,
    )
    parser.add_argument(
        "--error",
        help="Random error to apply to each TOA (us, default=1.0)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--addnoise",
        action="store_true",
        default=False,
        help="Actually add in random noise, or just populate the column",
    )
    parser.add_argument(
        "--fuzzdays",
        help="Standard deviation of 'fuzz' distribution (jd) (default: 0.0)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--plot", help="Plot residuals", action="store_true", default=False
    )
    parser.add_argument(
        "--format", help="The format of out put .tim file.", default="TEMPO2"
    )
    args = parser.parse_args(argv)

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    out_format = args.format
    error = args.error * u.microsecond

    if args.inputtim is None:
        log.info("Generating uniformly spaced TOAs")
        ts = pint.simulation.make_fake_toas_uniform(
            startMJD=args.startMJD,
            endMJD=args.startMJD + args.duration,
            ntoas=args.ntoa,
            model=m,
            obs=args.obs,
            error=error,
            freq=np.atleast_1d(args.freq) * u.MHz,
            fuzz=args.fuzzdays * u.d,
            add_noise=args.addnoise,
        )
    else:
        log.info("Reading initial TOAs from {0}".format(args.inputtim))
        ts = pint.simulation.make_fake_toas_fromtim(
            args.inputtim,
            model=m,
            obs=args.obs,
            error=error,
            freq=np.atleast_1d(args.freq) * u.MHz,
            fuzz=args.fuzzdays * u.d,
            add_noise=args.addnoise,
        )

    # Write TOAs to a file
    ts.write_TOA_file(args.timfile, name="fake", format=out_format)

    if args.plot:
        # This should be a very boring plot with all residuals flat at 0.0!
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support

        quantity_support()

        r = pint.residuals.Residuals(ts, m)
        plt.errorbar(
            ts.get_mjds(),
            r.calc_time_resids(calctype="taylor").to(u.us),
            yerr=ts.get_errors().to(u.us),
            fmt=".",
        )
        plt.xlabel("MJD")
        plt.ylabel("Residual (us)")
        plt.grid(True)
        plt.show()
