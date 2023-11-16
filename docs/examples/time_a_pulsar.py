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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Time a pulsar
#
# This notebook walks through a simple pulsar timing session, as one might do with TEMPO/TEMPO2: load a `.par` file, load a `.tim` file, do a fit, plot the residuals before and after. This one also displays various additional information you might find useful, and also ignores but then plots TOAs with large uncertainties. Similar code is available as a standalone script at [fit_NGC6440E.py](https://github.com/nanograv/PINT/blob/master/docs/examples/fit_NGC6440E.py)

# %%
import astropy.units as u
import matplotlib.pyplot as plt

import pint.fitter
from pint.models import get_model_and_toas
from pint.residuals import Residuals
import pint.logging

pint.logging.setup(level="INFO")

# %% [markdown]
#
# We want to load a parameter file and some TOAs. For the purposes of this notebook, we'll load in ones that are included with PINT; the `pint.config.examplefile()` calls return the path to where those files are in the PINT distribution. If you wanted to use your own files, you would probably know their filenames and could just set `parfile="myfile.par"` and `timfile="myfile.tim"`.

# %%
import pint.config

parfile = pint.config.examplefile("NGC6440E.par")
timfile = pint.config.examplefile("NGC6440E.tim")

# %% [markdown]
# Let's load the par and tim files. We could load them separately with the `get_model` and `get_TOAs` functions, but the parfile may contain information about how to interpret the TOAs, so it is convenient to load the two together so that the TOAs take into account details in the par file.

# %%
m, t_all = get_model_and_toas(parfile, timfile)
m

# %% [markdown]
# There are many messages here. As a rule messages marked `INFO` can safely be ignored, they are simply informational; take a look at them if something unexpected happens. Messages marked `WARNING` or `ERROR` are more serious. (These messages are emitted by the python `logger` module and can be suppressed or written to a log file if they are annoying.)
#
# Let's just print out a quick summary.

# %%
t_all.print_summary()

# %%
rs = Residuals(t_all, m).phase_resids
xt = t_all.get_mjds()
plt.figure()
plt.plot(xt, rs, "x")
plt.title(f"{m.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()

# %% [markdown]
# We could proceed immediately to fitting the par file, but some of those uncertainties seem a little large. Let's discard the data points with uncertainties $>30\,\mu\text{s}$ - uncertainty estimation is not always reliable when the signal-to-noise is low.

# %%
error_ok = t_all.table["error"] <= 30 * u.us
t = t_all[error_ok]
t.print_summary()

# %%
rs = Residuals(t, m).phase_resids
xt = t.get_mjds()
plt.figure()
plt.plot(xt, rs, "x")
plt.title(f"{m.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()

# %% [markdown]
# Now let's fit the par file to the residuals, using the `auto` function to pick the right fitter for our data.

# %%
f = pint.fitter.Fitter.auto(t, m)
f.fit_toas()

# %%
# Print some basic params
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))

# %%
# Show the parameter correlation matrix
corm = f.get_parameter_correlation_matrix(pretty_print=True)

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
plt.title(f"{m.PSR.value} Post-Fit Timing Residuals")
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
plt.title(f"{m.PSR.value} Post-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
plt.legend(loc="upper left")

# %%
plt.show()

# %%
f.model.write_parfile("/tmp/output.par", "wt")
print(f.model.as_parfile())
