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
# # Demonstrate phase offset
#
# This notebook is primarily designed to operate as a plain `.py` script.
# You should be able to run the `.py` script that occurs in the
# `docs/examples/` directory in order to carry out a simple fit of a
# timing model to some data. You should also be able to run the notebook
# version as it is here (it may be necessary to `make notebooks` to
# produce a `.ipynb` version using `jupytext`).

# %%
from pint.models import get_model_and_toas, PhaseOffset
from pint.residuals import Residuals
from pint.config import examplefile
from pint.fitter import DownhillWLSFitter

import matplotlib.pyplot as plt
from astropy.visualization import quantity_support

quantity_support()

# %%
# Read the TOAs and the model
parfile = examplefile("J1028-5819-example.par")
timfile = examplefile("J1028-5819-example.tim")
model, toas = get_model_and_toas(parfile, timfile)

# %%
# Create a Residuals object
res = Residuals(toas, model)

# %%
# By default, the residuals are mean-subtracted.
resids1 = res.calc_phase_resids().to("")

# We can disable mean subtraction by setting `subtract_mean` to False.
resids2 = res.calc_phase_resids(subtract_mean=False).to("")

# %%
# Let us plot the residuals with and without mean subtraction.
# In the bottom plot, there is clearly an offset between the two cases
# although it is not so clear in the top plot.

mjds = toas.get_mjds()
errors = toas.get_errors() * model.F0.quantity

plt.subplot(211)
plt.errorbar(mjds, resids1, errors, ls="", marker="x", label="Mean subtracted")
plt.errorbar(mjds, resids2, errors, ls="", marker="x", label="Not mean subtracted")
plt.xlabel("MJD")
plt.ylabel("Phase residuals")
plt.axhline(0, ls="dotted", color="grey")
plt.legend()

plt.subplot(212)
plt.plot(mjds, resids2 - resids1, ls="", marker="x")
plt.xlabel("MJD")
plt.ylabel("Phase residual difference")
plt.show()

# %%
# This phase offset that gets subtracted implicitly can be computed
# using the `calc_phase_mean` function. There is also a similar function
# `calc_time_mean` for time offsets.

implicit_offset = res.calc_phase_mean().to("")
print("Implicit offset = ", implicit_offset)

# %%
# Now let us look at the design matrix.
T, Tparams, Tunits = model.designmatrix(toas)
print("Design matrix params :", Tparams)

# The implicit offset is represented as "Offset".

# %%
# We can explicitly fit for this offset using the "PHOFF" parameter.
# This is available in the "PhaseOffset" component

po = PhaseOffset()
model.add_component(po)
assert hasattr(model, "PHOFF")

# %%
# Let us fit this now.

model.PHOFF.frozen = False
ftr = DownhillWLSFitter(toas, model)
ftr.fit_toas()
print(
    f"PHOFF fit value = {ftr.model.PHOFF.value} +/- {ftr.model.PHOFF.uncertainty_value}"
)

# This is consistent with the implicit offset we got earlier.

# %%
# Let us plot the post-fit residuals.

mjds = ftr.toas.get_mjds()
errors = ftr.toas.get_errors() * model.F0.quantity
resids = ftr.resids.calc_phase_resids().to("")

plt.errorbar(mjds, resids, errors, ls="", marker="x", label="After fitting PHOFF")
plt.xlabel("MJD")
plt.ylabel("Phase residuals")
plt.axhline(0, ls="dotted", color="grey")
plt.legend()
plt.show()

# %%
# Let us compute the phase residual mean again.
phase_mean = ftr.resids.calc_phase_mean().to("")
print("Phase residual mean = ", phase_mean)

# i.e., we have successfully gotten rid of the offset by fitting PHOFF.

# %%
# Now let us look at the design matrix again.
T, Tparams, Tunits = model.designmatrix(toas)
print("Design matrix params :", Tparams)

# The explicit offset "PHOFF" has replaced the implicit "Offset".
