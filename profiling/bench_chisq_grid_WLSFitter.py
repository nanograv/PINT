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
    include_gps=False,
    bipm_version="BIPM2015",
    include_bipm=False,
)

# Load model
thankmod = pint.models.get_model("J0740+6620.par")

# Fit one time
thankftr = pint.fitter.WLSFitter(toas=thanktoas, model=thankmod)
chisq = thankftr.fit_toas()

# Fit 3 x 3 grid of chisq values
n = 3
sini_grid = np.sin(np.linspace(86.25 * u.deg, 88.5 * u.deg, n))
m2_grid = np.linspace(0.2 * u.solMass, 0.30 * u.solMass, n)
thankftr_chi2grid = grid_chisq(thankftr, ("M2", "SINI"), (m2_grid, sini_grid), ncpu=1)

print()
print(f"Number of TOAs: {str(thanktoas.ntoas)}")
print(f"Grid size of parameters: {n}x{n}")
print("Number of fits: 1")
print()

print("Done")
