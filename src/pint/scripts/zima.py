#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""PINT-based tool for making simulated TOAs."""

import astropy.units as u
import numpy as np

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

import pint
import pint.fitter
import pint.models
import pint.simulation
import pint.residuals

__all__ = ["main"]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="PINT tool for simulating TOAs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="MJD of first fake TOA",
        type=float,
        default=56000.0,
    )
    parser.add_argument(
        "--ntoa", help="Number of fake TOAs to generate", type=int, default=100
    )
    parser.add_argument(
        "--duration", help="Span of TOAs to generate (days)", type=float, default=400.0
    )
    parser.add_argument("--obs", help="Observatory code", default="GBT")
    parser.add_argument(
        "--freq",
        help="Frequency for TOAs (MHz)",
        nargs="+",
        type=float,
        default=1400.0,
    )
    parser.add_argument(
        "--multifreq",
        help="Simulate multiple frequency TOAs per epoch",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--error",
        help="Random error to apply to each TOA (us)",
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
        "--addcorrnoise",
        action="store_true",
        default=False,
        help="Add in a correlated noise realization if it's present in the model",
    )
    parser.add_argument(
        "--wideband",
        action="store_true",
        default=False,
        help="Add DM information to simulated TOAs. Generates wideband toas.",
    )
    parser.add_argument(
        "--dmerror",
        help="Random error to apply to simulated DM measurements (dmu)",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--fuzzdays",
        help="Standard deviation of 'fuzz' distribution (jd)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--plot", help="Plot residuals", action="store_true", default=False
    )
    parser.add_argument(
        "--format", help="The format of output .tim file.", default="TEMPO2"
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
            add_correlated_noise=args.addcorrnoise,
            wideband=args.wideband,
            wideband_dm_error=args.dmerror * pint.dmu,
            multi_freqs_in_epoch=args.multifreq,
        )
    else:
        log.info(f"Reading initial TOAs from {args.inputtim}")
        ts = pint.simulation.make_fake_toas_fromtim(
            args.inputtim,
            model=m,
            add_noise=args.addnoise,
            add_correlated_noise=args.addcorrnoise,
        )

    # Write TOAs to a file
    ts.write_TOA_file(args.timfile, name="fake", format=out_format)

    if args.plot:
        plot_simulated_toas(ts, m)


def plot_simulated_toas(ts, m):
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
