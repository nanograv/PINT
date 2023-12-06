# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PINT Noise Fitting Examples

# %%
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.logging import setup as setup_log
from pint.fitter import Fitter

import numpy as np
from io import StringIO
from astropy import units as u
from matplotlib import pyplot as plt

# %%
setup_log(level="WARNING")

# %% [markdown]
# ## Fitting for EFAC and EQUAD

# %%
# Let us begin by simulating a dataset with an EFAC and an EQUAD.
# Note that the EFAC and the EQUAD are set as fit parameters ("1").
par = """
    PSR             TEST1
    RAJ             05:00:00    1
    DECJ            15:00:00    1
    PEPOCH          55000
    F0              100         1
    F1              -1e-15      1
    EFAC tel gbt    1.3         1
    EQUAD tel gbt   1.1         1
    TZRMJD          55000
    TZRFRQ          1400
    TZRSITE         gbt
    EPHEM           DE440
    CLOCK           TT(BIPM2019)
    UNITS           TDB
"""

m = get_model(StringIO(par))

ntoas = 200

# EFAC and EQUAD cannot be measured separately if all TOA uncertainties
# are the same. So we must set a different toa uncertainty for each TOA.
# This is how it is in real datasets anyway.
toaerrs = np.random.uniform(0.5, 2, ntoas) * u.us

t = make_fake_toas_uniform(
    startMJD=54000,
    endMJD=56000,
    ntoas=ntoas,
    model=m,
    obs="gbt",
    error=toaerrs,
    add_noise=True,
    include_bipm=True,
    include_gps=True,
)

# %%
# Now create the fitter. The `Fitter.auto()` function creates a
# Downhill fitter. Noise parameter fitting is only available in
# Downhill fitters.
ftr = Fitter.auto(t, m)

# %%
# Now do the fitting.
ftr.fit_toas()

# %%
# Print the post-fit model. We can see that the EFAC and EQUAD have been
# and the uncertainties are listed.
print(ftr.model)

# %%
# Let us plot the injected and measured noise parameters together to
# compare them.
plt.scatter(m.EFAC1.value, m.EQUAD1.value, label="Injected", marker="o", color="blue")
plt.errorbar(
    ftr.model.EFAC1.value,
    ftr.model.EQUAD1.value,
    xerr=ftr.model.EFAC1.uncertainty_value,
    yerr=ftr.model.EQUAD1.uncertainty_value,
    marker="+",
    label="Measured",
    color="red",
)
plt.xlabel("EFAC_tel_gbt")
plt.ylabel("EQUAD_tel_gbt (us)")
plt.legend()
plt.show()

# %% [markdown]
# ## Fitting for ECORRs

# %%
# Note the explicit offset (PHOFF) in the par file below.
# Implicit offset subtraction is typically not accurate enough when
# ECORR (or any other type of correlated noise) is present.
# i.e., PHOFF should be a free parameter when ECORRs are being fit.
par = """
    PSR             TEST2
    RAJ             05:00:00    1
    DECJ            15:00:00    1
    PEPOCH          55000
    F0              100         1
    F1              -1e-15      1
    PHOFF           0           1
    EFAC tel gbt    1.3         1
    ECORR tel gbt   1.1         1
    TZRMJD          55000
    TZRFRQ          1400
    TZRSITE         gbt
    EPHEM           DE440
    CLOCK           TT(BIPM2019)
    UNITS           TDB
"""

m = get_model(StringIO(par))

# ECORRs only apply when there are multiple TOAs per epoch.
# This can be simulated by providing multiple frequencies and
# setting the `multi_freqs_in_epoch` option. The `add_correlated_noise`
# option should also be set because correlated noise components
# are not simulated by default.

ntoas = 500
toaerrs = np.random.uniform(0.5, 2, ntoas) * u.us
freqs = np.linspace(1300, 1500, 4) * u.MHz

t = make_fake_toas_uniform(
    startMJD=54000,
    endMJD=56000,
    ntoas=ntoas,
    model=m,
    obs="gbt",
    error=toaerrs,
    freq=freqs,
    add_noise=True,
    add_correlated_noise=True,
    include_bipm=True,
    include_gps=True,
    multi_freqs_in_epoch=True,
)

# %%
ftr = Fitter.auto(t, m)

# %%
ftr.fit_toas()

# %%
print(ftr.model)

# %%
# Let us plot the injected and measured noise parameters together to
# compare them.
plt.scatter(m.EFAC1.value, m.ECORR1.value, label="Injected", marker="o", color="blue")
plt.errorbar(
    ftr.model.EFAC1.value,
    ftr.model.ECORR1.value,
    xerr=ftr.model.EFAC1.uncertainty_value,
    yerr=ftr.model.ECORR1.uncertainty_value,
    marker="+",
    label="Measured",
    color="red",
)
plt.xlabel("EFAC_tel_gbt")
plt.ylabel("ECORR_tel_gbt (us)")
plt.legend()
plt.show()
