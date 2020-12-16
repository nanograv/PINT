# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Time a pulsar
#
# This notebook walks through a simple pulsar timing session, as one might do with TEMPO/TEMPO2: load a `.par` file, load a `.tim` file, do a fit, plot the residuals before and after. This one also displays various additional information you might find useful, and also ignores but then plots TOAs with large uncertainties. Similar code is available as a standalone script at [`fit_NGC6440E.py`](https://github.com/nanograv/PINT/blob/master/docs/examples/fit_NGC6440E.py)

# %%
import astropy.units as u
import matplotlib.pyplot as plt

import pint.fitter
from pint.models import get_model_and_toas
from pint.residuals import Residuals
from pint.toa import get_TOAs

# %%
parfile = "NGC6440E.par"
timfile = "NGC6440E.tim"

# %%
m, t_all = get_model_and_toas(parfile, timfile)
m

# %%
t_all.print_summary()

# %% [markdown]
# There are many messages here. As a rule messages marked `INFO` can safely be ignored, they are simply informational; take a look at them if something unexpected happens. Messages marked `WARNING` or `ERROR` are more serious. (These messages are emitted by the python `logger` module and can be suppressed or written to a log file if they are annoying.)

# %% [markdown]
# Let's discard the data points with uncertainties $>30\,\mu\text{s}$ - uncertainty estimation is not always reliable when the signal-to-noise is low.

# %%
error_ok = t_all.table["error"] <= 30 * u.us
t = t_all[error_ok]
t.print_summary()

# %%
# These are pre-fit residuals
rs = Residuals(t, m).phase_resids
xt = t.get_mjds()
plt.figure()
plt.plot(xt, rs, "x")
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()

# %%
f = pint.fitter.WLSFitter(t, m)
# f = pint.fitter.PowellFitter(t, m)
f.fit_toas()
# f = pint.fitter.GLSFitter(t, m)
# f.fit_toas(full_cov=True)

# %%
# Print some basic params
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))

# %%
# Show the parameter correlation matrix
corm = f.get_correlation_matrix(pretty_print=True)

# %%
f.print_summary()

# %%
plt.figure()
plt.errorbar(
    xt.value,
    f.resids.time_resids.to(u.us).value,
    t.get_errors().to(u.us).value,
    fmt="x",
)
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()

# %%
t_bad = t_all[~error_ok]
r_bad = Residuals(t_bad, f.model)
plt.figure()
plt.errorbar(
    xt.value,
    f.resids.time_resids.to(u.us).value,
    t.get_errors().to(u.us).value,
    fmt="x",
    label="used in fit",
)
plt.errorbar(
    t_bad.get_mjds().value,
    r_bad.time_resids.to(u.us).value,
    t_bad.get_errors().to(u.us).value,
    fmt="x",
    label="bad data",
)
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
plt.legend(loc="upper left")

# %%
plt.show()

# %%
