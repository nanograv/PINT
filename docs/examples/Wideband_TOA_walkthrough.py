# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Wideband TOA fitting
#
# Traditional pulsar timing involved measuring only the arrival time of each pulse. But as receivers have covered wider and wider contiguous bandwidths, it became necessary to generate many TOAs for each time interval, covering different subbands. This frequency coverage allowed better handling of changing dispersion measures, but resulted in a large number of TOAs and had certain limitations. A new approach measures the pulse arrival time and the dispersion measure simultaneously from a frequency-resolved data cube. This produces TOAs, each of which has an associated dispersion measure and uncertainty. Working with this data requires different handling from PINT. This notebook demonstrates that.

# %%
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support

from pint.fitter import Fitter
from pint.models import get_model_and_toas
import pint.config
import pint.logging

# setup logging
pint.logging.setup(level="INFO")

quantity_support()

# %% [markdown]
# ## Set up your inputs

# %%
model, toas = get_model_and_toas(
    pint.config.examplefile("J1614-2230_NANOGrav_12yv3.wb.gls.par"),
    pint.config.examplefile("J1614-2230_NANOGrav_12yv3.wb.tim"),
)

# %% [markdown]
# The DM and its uncertainty are recorded as flags, `pp_dm` and `pp_dme` on the TOAs that have them, They are not currently available as Columns in the Astropy object. On the other hand, it is not necessary that every observation have a measured DM.
#
# (The name, `pp_dm`, refers to the fact that they are obtained using "phase portraits", like profiles but in one more dimension.)

# %%
print(open(toas.filename).readlines()[-1])

# %%
toas.table[-1]

# %%
toas.table["flags"][0]

# %% [markdown]
# ## Do the fit
#
# As before, but now we need a fitter adapted to wideband TOAs. The function `Fitter.auto()` will examine the model and choose an appropriate one.

# %%
fitter = Fitter.auto(toas, model)

# %%
fitter.fit_toas()

# %% [markdown]
# ## What is new, compared to narrowband fitting?

# %% [markdown]
# ### Residual objects combine TOA and time data

# %%
type(fitter.resids)

# %% [markdown]
# #### If we look into the resids attribute, it has two independent Residual objects.

# %%
fitter.resids.toa, fitter.resids.dm

# %% [markdown]
# #### Each of them can be used independently
#
# * Time residual

# %%
time_resids = fitter.resids.toa.time_resids
plt.errorbar(
    toas.get_mjds().value,
    time_resids.to_value(u.us),
    yerr=toas.get_errors().to_value(u.us),
    fmt="x",
)
plt.ylabel("us")
plt.xlabel("MJD")

# %%
# Time RMS
print(fitter.resids.toa.rms_weighted())
print(fitter.resids.toa.chi2)

# %% [markdown]
# * DM residual

# %%
dm_resids = fitter.resids.dm.resids
dm_error = fitter.resids.dm.get_data_error()
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt="x")
plt.ylabel("pc/cm^3")
plt.xlabel("MJD")

# %%
# DM RMS
print(fitter.resids.dm.rms_weighted())
print(fitter.resids.dm.chi2)

# %% [markdown]
# #### However, in the combined residuals, one can access rms and chi2 as well

# %%
print(fitter.resids.rms_weighted())
print(fitter.resids.chi2)

# %% [markdown]
# #### The initial residuals is also a combined residual object

# %%
time_resids = fitter.resids_init.toa.time_resids
plt.errorbar(
    toas.get_mjds().value,
    time_resids.to_value(u.us),
    yerr=toas.get_errors().to_value(u.us),
    fmt="x",
)
plt.ylabel("us")
plt.xlabel("MJD")

# %%
dm_resids = fitter.resids_init.dm.resids
dm_error = fitter.resids_init.dm.get_data_error()
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt="x")
plt.ylabel("pc/cm^3")
plt.xlabel("MJD")

# %% [markdown]
# ### Matrices
#
# We're now fitting a mixed set of data, so the matrices used in fitting now have different units in different parts, and some care is needed to keep track of which part goes where.

# %% [markdown]
# #### Design Matrix are combined

# %%
d_matrix, labels, units = fitter.get_designmatrix()

# %%
print("Number of TOAs:", toas.ntoas)
print("Number of DM measurments:", len(fitter.resids.dm.dm_data))
print("Number of fit params:", len(fitter.model.free_params))
print("Shape of design matrix:", d_matrix.shape)

# %% [markdown]
# #### Covariance Matrix are combined

# %%
# c_matrix = fitter.get_noise_covariancematrix()

# %%
# print("Shape of covariance matrix:", c_matrix.shape)


# %% [markdown]
# NOTE the matrix are PINTMatrix object right now, here are the difference

# %% [markdown]
# If you want to access the matrix data

# %%
# print(d_matrix.matrix)

# %% [markdown]
# PINT matrix has labels that marks all the element in the matrix. It has the label name, index of range of the matrix, and the unit.

# %%
# print("labels for dimension 0:", d_matrix.labels[0])

# %%
