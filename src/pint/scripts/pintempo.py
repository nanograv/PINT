#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""Command-line interface for PINT

This is a command-line interface for PINT. It does *not* try to duplicate the
command line syntax for either TEMPO or Tempo2. (I never understood why I had to
specify '-f parfile' to those codes -- I mean, who runs TEMPO without a timing model?)

This is currently just a stub and should be added to and expanded, as desired.

"""
from __future__ import absolute_import, division, print_function

import argparse
import sys

import astropy.units as u
from astropy import log

import pint.fitter
import pint.models
import pint.residuals

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Command line interfact to PINT")
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="TOA file name")
    parser.add_argument(
        "--outfile", help="Output par file name (default=None)", default=None
    )
    parser.add_argument(
        "--plot", help="Plot residuals", action="store_true", default=False
    )
    parser.add_argument("--plotfile", help="Plot file name", default=None)
    args = parser.parse_args(argv)

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    log.warning(m.params)

    log.info("Reading TOAs")
    use_planets = False
    if m.PLANET_SHAPIRO.value:
        use_planets = True
    model_ephem = "DE421"
    if m.EPHEM is not None:
        model_ephem = m.EPHEM.value
    t = pint.toa.get_TOAs(args.timfile, planets=use_planets, ephem=model_ephem)
    prefit_resids = pint.residuals.Residuals(t, m).time_resids

    log.info("Fitting...")
    f = pint.fitter.WLSFitter(t, m)
    f.fit_toas()

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
        ax.set_title("%s Timing Residuals" % m.PSR.value)
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
