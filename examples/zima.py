#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""PINT-based tool for making simulated TOAs

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
from astropy.time import Time
import argparse

from astropy import log

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
    start = Time(56000.0,scale='utc',format='mjd')
    error = 1.0e-6*u.s
    freq = 1400.0*u.MHz
    scale = 'utc'

    times = np.linspace(0,duration.to(u.day).value,ntoa)*u.day + start
    
    tl = [toa.TOA(t,error=error, obs=obs, freq=freq,
                 scale=scale) for t in times]
                 
    ts = toa.TOAs(toalist=tl)
    
    