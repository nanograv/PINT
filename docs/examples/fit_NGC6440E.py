#! /usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: .env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demonstrate the use of PINT in a script
#
# This notebook is primarily designed to operate as a plain `.py` script.
# You should be able to run the `.py` script that occurs in the
# `docs/examples/` directory in order to carry out a simple fit of a
# timing model to some data. You should also be able to run the notebook
# version as it is here (it may be necessary to `make notebooks` to
# produce a `.ipynb` version using `jupytext`).

# %%
import os

import astropy.units as u

# This will change which output method matplotlib uses and may behave better on some machines
# import matplotlib
# matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import pint.fitter
import pint.residuals
import pint.toa
from pint.models import get_model_and_toas
import pint.logging
import os

# setup logging
pint.logging.setup(level="INFO")

# %%
import pint.config

parfile = pint.config.examplefile("NGC6440E.par")
timfile = pint.config.examplefile("NGC6440E.tim")
assert os.path.exists(parfile)
assert os.path.exists(timfile)

# %%
# Read the timing model and the TOAs
m, t = get_model_and_toas(parfile, timfile)

# %% [markdown]
# If we wanted to do things separately we could do
# ```python
# # Define the timing model
# m = get_model(parfile)
# # Read in the TOAs, using the solar system ephemeris and other things from the model
# t = pint.toa.get_TOAs(timfile, model=m)
# ```

# %% [markdown]
# If we wanted to select some subset of the TOAs, there are tools to do that. Most easily
# you make a new TOAs object containing the subset you care about (we will make but not use
# them):

# %% [markdown]
# Use every other TOA

# %%
t_every_other = t[::2]

# %% [markdown]
# Use only TOAs with errors < 30 us

# %%
t_small_errors = t[t.get_errors() < 30 * u.us]

# %% [markdown]
# Use only TOAs from the GBT (although this is all of them for this example)

# %%
t_gbt = t[t.get_obss() == "gbt"]

# %%
# Print a summary of the TOAs that we have
print(t.get_summary())

# %%
# These are pre-fit residuals
rs = pint.residuals.Residuals(t, m).phase_resids
xt = t.get_mjds()
plt.plot(xt, rs, "x")
plt.title(f"{m.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()
plt.show()

# %%
# Now do the fit
print("Fitting.")
f = pint.fitter.DownhillWLSFitter(t, m)
f.fit_toas()
# f = pint.fitter.DownhillGLSFitter(t, m)
# f.fit_toas(full_cov=True)

# %%
# Print some basic params
print("Best fit has reduced chi^2 of", f.resids.reduced_chi2)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))

# %%
# Show the parameter correlation matrix
corm = f.get_parameter_correlation_matrix(pretty_print=True)

# %%
print(f.get_summary())

# %%
plt.errorbar(
    xt.value,
    f.resids.time_resids.to_value(u.us),
    t.get_errors().to_value(u.us),
    fmt="x",
)
plt.title(f"{m.PSR.value} Post-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
plt.show()

# %%
f.model.write_parfile("/tmp/output.par", "wt")
print(f.model.as_parfile())
