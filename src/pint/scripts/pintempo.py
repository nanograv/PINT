#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""Command-line interface for PINT

This is a command-line interface for PINT. It does *not* try to duplicate the
command line syntax for either TEMPO or Tempo2. (I never understood why I had to
specify '-f parfile' to those codes -- I mean, who runs TEMPO without a timing model?)

This is currently just a stub and should be added to and expanded, as desired.
"""
import argparse
import sys

import astropy.units as u
from astropy import log

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

import pint.fitter
import pint.models
import pint.residuals

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Command line interface to PINT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="TOA file name")
    parser.add_argument(
        "--usepickle",
        help="Enable pickling of TOAs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--outfile", help="Output par file name (default=None)", default=None
    )
    parser.add_argument(
        "--plot", help="Plot residuals", action="store_true", default=False
    )
    parser.add_argument("--plotfile", help="Plot file name", default=None)
    parser.add_argument(
        "--gls", help="Fit using GLS fitter", action="store_true", default=False
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

    log.warning(m.params)

    log.info("Reading TOAs")
    t = pint.toa.get_TOAs(args.timfile, model=m, usepickle=args.usepickle)

    # turns pre-existing jump flags in t.table['flags'] into parameters in parfile
    m.jump_flags_to_params(t)

    # adds jump flags to t.table['flags'] for jump parameters already in parfile
    if "PhaseJump" in m.components:
        m.jump_params_to_flags(t)

    if m.TRACK.value == "-2":
        if "pn" in t.table.colnames:
            log.info("Already have pulse numbers from TOA flags.")
        else:
            log.info("Adding pulse numbers")
            t.compute_pulse_numbers(m)

    prefit_resids = pint.residuals.Residuals(t, m).time_resids

    log.info("Fitting...")
    if args.gls:
        f = pint.fitter.DownhillGLSFitter(t, m)
    else:
        f = pint.fitter.DownhillWLSFitter(t, m)
    f.fit_toas()
    f.update_model()
    f.model.CHI2.value = f.resids.chi2

    # Print fit summary
    print(
        "============================================================================"
    )
    f.print_summary()

    if args.plot:
        import matplotlib.pyplot as plt

        # Turn on support for plotting quantities
        from astropy.visualization import quantity_support

        quantity_support()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        xt = t.get_mjds()
        ax.errorbar(xt, prefit_resids.to(u.us), t.get_errors().to(u.us), fmt="o")
        ax.errorbar(xt, f.resids.time_resids.to(u.us), t.get_errors().to(u.us), fmt="x")
        ax.set_title(f"{m.PSR.value} Timing Residuals")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Residual (us)")
        ax.grid()
        if args.plotfile is not None:
            fig.savefig(args.plotfile)
        else:
            plt.show()

    if args.outfile is not None:
        fout = open(args.outfile, "w")
    else:
        fout = sys.stdout
        print("\nBest fit model is:")

    fout.write(f.model.as_parfile() + "\n")
    return 0
