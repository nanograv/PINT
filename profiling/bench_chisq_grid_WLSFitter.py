#!/usr/bin/env python

import pint.toa
import pint.models
import pint.fitter
from pint.gridutils import grid_chisq
import numpy as np
import astropy.units as u

# Get .tim file and par file from here:
# curl -O https://data.nanograv.org/static/data/J0740+6620.cfr+19.tim
# curl -O https://data.nanograv.org/static/data/J0740+6620.par

thanktoas = pint.toa.get_TOAs(
    "J0740+6620.cfr+19.tim",
    ephem="DE436",
    planets=True,
    usepickle=False,
    include_gps=True,
    bipm_version="BIPM2015",
    include_bipm=True,
)

# Load model
thankmod = pint.models.get_model("J0740+6620.par")

# Fit one time
thankftr = pint.fitter.WLSFitter(toas=thanktoas, model=thankmod)
chisq = thankftr.fit_toas()

# Fit 5 x 5 grid of chisq values
sini_grid = np.sin(np.linspace(86.25 * u.deg, 88.5 * u.deg, 5))
m2_grid = np.linspace(0.2 * u.solMass, 0.30 * u.solMass, 5)
thankftr_chi2grid = grid_chisq(thankftr, "M2", m2_grid, "SINI", sini_grid)

print("Done")
