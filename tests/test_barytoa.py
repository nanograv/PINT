#!/usr/bin/env python
import pint.toa
import pint.models
import pint.fitter
import numpy as np
import pint.residuals
import pint.models.model_builder as mb
import astropy.units as u
import os, sys

from pinttestdata import testdir, datadir

os.chdir(datadir)

# This par file has a very simple model in it
m = mb.get_model('slug.par')

# This .tim file has TOAs at the barycenter, and at infinite frequency
t = pint.toa.get_TOAs('slug.tim')

rs = pint.residuals.resids(t, m).time_resids

# Residuals should be less than 2.0 ms
assert rs.std() < 2.0*u.ms
