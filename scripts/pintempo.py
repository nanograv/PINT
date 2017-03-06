#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""Command-line interface for PINT

This is a command-line interface for PINT. It does *not* try to duplicate the
command line syntax for either TEMPO or Tempo2. (I never understood why I had to 
specify '-f parfile' to those codes -- I mean, who runs TEMPO without a timing model?)

This is currently just a stub and should be added to and expanded, as desired.

"""
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.fitter
import pint.residuals
import astropy.units as u
import matplotlib.pyplot as plt
import argparse

from astropy import log

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Command line interfact to PINT")
    parser.add_argument("parfile",help="par file to read model from")
    parser.add_argument("timfile",help="TOA file name")
    parser.add_argument("--outfile",help="Output figure file name (default=None)", default=None)
    parser.add_argument("--plot",help="Plot residuals",action="store_true",default=False)
    args = parser.parse_args()

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    log.info("Reading TOAs")
    t = pint.toa.get_TOAs(args.timfile)
    prefit_resids = pint.residuals.resids(t, m).time_resids

    log.info("Fitting...")
    f = pint.fitter.WlsFitter(t, m)
    f.fit_toas()

    # Print some basic params
    print( "Best fit has reduced chi^2 of", f.resids.chi2_reduced)
    print( "RMS in phase is", f.resids.phase_resids.std())
    print( "RMS in time is", f.resids.time_resids.std().to(u.us))

    if args.plot:
        xt = t.get_mjds()
        plt.errorbar(xt,prefit_resids.to(u.us).value,
                    t.get_errors().to(u.us).value, fmt='o')
        plt.errorbar(xt,
                 f.resids.time_resids.to(u.us).value,
                 t.get_errors().to(u.us).value, fmt='x')
        plt.title("%s Timing Residuals" % m.PSR.value)
        plt.xlabel('MJD')
        plt.ylabel('Residual (us)')
        plt.grid()
        plt.show()

    if args.outfile is not None:
        fout = file(args.outfile,"w")
    else:
        fout = sys.stdout
        print("\nBest fit model is:")

    fout.write(f.model.as_parfile()+"\n")


