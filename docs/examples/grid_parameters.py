# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit data over a grid of parameters
# 1. Load a data-set
# 2. Find the best-fit
# 3. Determine $\chi^2$ of a 1D grid of parameters
# 4. Determine $\chi^2$ over a grid of 2 parameters
# 5. Plot the contours of $\chi^2$ over that grid, and compare the 1D and 2D confidence regions

# %% tags=[]
import astropy.units as u
import numpy as np
import copy

import pint.config
import pint.gridutils
import pint.models.parameter as param
import pint.residuals
from pint.fitter import WLSFitter
from pint.models.model_builder import get_model, get_model_and_toas
from pint.toa import get_TOAs
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support

quantity_support()
import scipy.stats

# %matplotlib inline

# %%
# Load in a basic dataset
parfile = pint.config.examplefile("NGC6440E.par")
timfile = pint.config.examplefile("NGC6440E.tim")
m, t = get_model_and_toas(parfile, timfile)

f = WLSFitter(t, m)
# find the best-fit
f.fit_toas()
bestfit = f.resids.chi2

# %% [markdown]
# What are the free parameters?

# %%
f.model.free_params

# %% [markdown]
# So we were fitting for RA, Dec, $F0$, $F1$, and $DM$.
#
# We'll do something like 3-sigma around the best-fit value of $F0$, fitting for RA, Dec, $F1$ and $DM$ at each grid point.

# %% [markdown]
# ## 1D Grid

# %%
F0 = np.linspace(
    f.model.F0.quantity - 3 * f.model.F0.uncertainty,
    f.model.F0.quantity + 3 * f.model.F0.uncertainty,
    25,
)

# %%
# Do a 1D "grid"  Make sure that the parameters are supplied as tuples with length 1
chi2_F0, _ = pint.gridutils.grid_chisq(f, ("F0",), (F0,))

# %%
# We can now just do a quick plot to look at the results
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(F0 - f.model.F0.quantity, chi2_F0, label="$\chi^2$ curve")
ax.set_xlabel("$\Delta F_0$ (Hz)", fontsize=24)
ax.set_ylabel("$\chi^2$", fontsize=24)

# %%
# We can go a little further.
# For instance, we can find the uncertainty by looking where Delta chi^2=1
F0_delta_chi2 = np.interp(
    1,
    chi2_F0[F0 >= f.model.F0.quantity] - chi2_F0.min(),
    F0[F0 >= f.model.F0.quantity].astype(np.float64),
)

# %%
fig, ax = plt.subplots(figsize=(16, 9))
# just plot the values offset from the best-fit values
ax.plot(F0 - f.model.F0.quantity, chi2_F0, label="$\chi^2$ curve")
ax.plot(0, chi2_F0.min(), "ro", markersize=12, label="Minimum $\chi^2$")
ax.plot((f.model.F0.uncertainty) * np.ones(2), [59, 69], "r:", label="Uncertainty")
ax.plot((-f.model.F0.uncertainty) * np.ones(2), [59, 69], "r:")
ax.plot(
    [
        F0_delta_chi2 - f.model.F0.quantity,
        F0_delta_chi2 - f.model.F0.quantity,
        -(F0_delta_chi2 - f.model.F0.quantity),
        -(F0_delta_chi2 - f.model.F0.quantity),
    ],
    [chi2_F0.min(), chi2_F0.min() + 1, chi2_F0.min() + 1, chi2_F0.min()],
    "g--",
    label="$\Delta \chi^2=1$",
)
ax.set_xlabel("$\Delta F_0$ (Hz)", fontsize=24)
ax.set_ylabel("$\chi^2$", fontsize=24)
ax.set_ylim([59, 69])
ax.legend()

# %% [markdown]
# From the above you can see that the calculated uncertainties on $F0$ from the model fit agree with where $\Delta \chi^2=1$, which is what we would expect for a single parameter.

# %% [markdown]
# ## 2D Grid

# %% [markdown]
# Now, compute a 2D grid of $\chi^2(F0,F1)$, so we'll also set up a grid of $F1$ to search over

# %% tags=[]
F1 = np.linspace(
    f.model.F1.quantity - 3 * f.model.F1.uncertainty,
    f.model.F1.quantity + 3 * f.model.F1.uncertainty,
    27,
)
chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1))

# %% [markdown]
# We want to plot contour levels appropriate for the joint confidence contours with 2 parameters.  This is discussed many places (e.g., https://ned.ipac.caltech.edu/level5/Wall2/Wal3_4.html, or Chapter 15.6 of Numerical Recipes in C - in particular look at Figure 15.6.4).
# Rather than look up the values in a table we will compute them ourselves.
#
# The goal is to find $x$ such that the CDF of the $\chi^2_\nu$ (for $\nu=2$ parameters) distribution evaluated at $x$ is equal to the desired confidence interval.
#
# We want 1, 2, and 3 $\sigma$ confidence intervals.  So first we determine what the CDFs of the normal distribution are at those values.
#
# We then get a $\chi^2_2$ random variable, and interpolate to find the values of $x$ (which are our desired confidence intervals).
#
# We can check these against the values in the linked table (or elsewhere).

# %% tags=[]
# 1, 2, and 3 sigma confidence limits
nsigma = np.arange(1, 4)
# these are the CDFs going from -infinity to nsigma.  So subtract away 0.5 and double for the 2-sided values
CIs = (scipy.stats.norm().cdf(nsigma) - 0.5) * 2
print(f"Confidence intervals for {nsigma} sigma: {CIs}")
# chi^2 random variable for 2 parameters
rv = scipy.stats.chi2(2)
# the ppf = Percent point function is the inverse of the CDF
contour_levels = rv.ppf(CIs)
print(f"Contour levels for {nsigma} sigma and 2 parameters: {contour_levels}")

# %% [markdown]
# Let's repeat that for a single parameter for comparison

# %%
# 1, 2, and 3 sigma confidence limits
nsigma = np.arange(1, 4)
# these are the CDFs going from -infinity to nsigma.  So subtract away 0.5 and double for the 2-sided values
CIs = (scipy.stats.norm().cdf(nsigma) - 0.5) * 2
print(f"Confidence intervals for {nsigma} sigma: {CIs}")
# chi^2 random variable for 2 parameters
rv = scipy.stats.chi2(1)
contour_levels_1param = rv.ppf(CIs)
print(f"Contour levels for {nsigma} sigma and 1 parameter: {contour_levels_1param}")

# %%
fig, ax = plt.subplots(figsize=(16, 9))
# just plot the values offset from the best-fit values
twod = ax.contour(
    F0 - f.model.F0.quantity,
    F1 - f.model.F1.quantity,
    chi2grid - bestfit,
    levels=contour_levels,
    colors="b",
)
oned = ax.contour(
    F0 - f.model.F0.quantity,
    F1 - f.model.F1.quantity,
    chi2grid - bestfit,
    levels=contour_levels_1param,
    colors="g",
    linestyles="--",
)
ax.errorbar(
    0, 0, xerr=f.model.F0.uncertainty.value, yerr=f.model.F1.uncertainty.value, fmt="ro"
)
ax.set_xlabel("$\Delta F_0$ (Hz)", fontsize=24)
ax.set_ylabel("$\Delta F_1$ (Hz/s)", fontsize=24)
twod_artists, _ = twod.legend_elements()
oned_artists, _ = oned.legend_elements()
ax.legend(
    [twod_artists[0], oned_artists[0]],
    ["Joint 2D fit", "Single-parameter Fit"],
    fontsize=18,
)

# %% [markdown]
# It's pretty clear that $F0$ and $F1$ are highly (anti)correlated.  This can be improved by chosing **PEPOCH**.
#
# We can look at the correlation matrix explicitly:

# %%
f.parameter_correlation_matrix

# %% [markdown]
# We can see that the $(F0,F1)$ is $-0.8$, which agrees (qualitatively) with what we see above.

# %% [markdown]
# It's also clear that the joint confidence region (solid blue contours) are significantly bigger than the single parameter region (green dashed contours).

# %% [markdown]
# ## Changing PEPOCH

# %% [markdown]
# To have minimal covariance between $F0$ and $F1$, we want PEPOCH to be close to the mean time of the TOAs (ideally the weighted mean).  Is it close now?

# %%
print(f"PEPOCH: {f.model.PEPOCH}")
print(f"Mean TOA: {f.toas.get_mjds().mean()}")

# %% [markdown]
# So they differ by 141 days. Let's change that.

# %%
# keep a copy of our old fit for reference
old_f = copy.deepcopy(f)

# %%
f.model.change_pepoch(f.toas.get_mjds().mean())
f.fit_toas()
new_bestfit = f.resids.chi2
print(f"Old chi^2: {bestfit}, new chi^2: {new_bestfit}")
print(
    f"Old F0: {old_f.model.F0.quantity}, new F0: {f.model.F0.quantity}, difference {(f.model.F0.quantity - old_f.model.F0.quantity)/f.model.F0.uncertainty} sigma"
)
print(
    f"Old F1: {old_f.model.F1.quantity}, new F1: {f.model.F1.quantity}, difference {(f.model.F1.quantity - old_f.model.F1.quantity)/f.model.F1.uncertainty} sigma"
)

# %% [markdown]
# We see that the overall fit quality ($\chi^2$) is the same. That's good.
#
# $F1$ doesn't change.  That is expected.  But $F0$ does.  However, that's mostly just from changing epoch to redefine when $F0$ is evaluated.  We can check this by looking at what change we expect for $F0$:

# %%
print(
    f"Predicted new F0 - fitted new F0 = {(old_f.model.F0.quantity + (f.model.PEPOCH.quantity - old_f.model.PEPOCH.quantity)*f.model.F1.quantity - f.model.F0.quantity)/f.model.F0.uncertainty} sigma"
)

# %% [markdown]
# So the change is pretty close to what we expect.

# %%
# update the grids
new_F0 = np.linspace(
    f.model.F0.quantity - 3 * f.model.F0.uncertainty,
    f.model.F0.quantity + 3 * f.model.F0.uncertainty,
    len(F0),
)
new_F1 = np.linspace(
    f.model.F1.quantity - 3 * f.model.F1.uncertainty,
    f.model.F1.quantity + 3 * f.model.F1.uncertainty,
    len(F1),
)

# %%
new_chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0", "F1"), (new_F0, new_F1))

# %% [markdown]
# Plot the new and old contours

# %%
fig, ax = plt.subplots(figsize=(16, 9))
# just plot the values offset from the best-fit values
old = ax.contour(
    F0 - old_f.model.F0.quantity,
    F1 - old_f.model.F1.quantity,
    chi2grid - bestfit,
    levels=contour_levels,
    colors="b",
)
new = ax.contour(
    new_F0 - f.model.F0.quantity,
    new_F1 - f.model.F1.quantity,
    new_chi2grid - bestfit,
    levels=contour_levels,
    colors="k",
    linestyles=":",
)
ax.errorbar(
    0, 0, xerr=f.model.F0.uncertainty.value, yerr=f.model.F1.uncertainty.value, fmt="ro"
)
ax.set_xlabel("$\Delta F_0$ (Hz)", fontsize=24)
ax.set_ylabel("$\Delta F_1$ (Hz/s)", fontsize=24)
old_artists, _ = old.legend_elements()
new_artists, _ = new.legend_elements()
ax.legend([old_artists[0], new_artists[0]], ["Old fit", "New Fit"], fontsize=18)

# %% [markdown]
# The new contours (the black) look a lot more "orthogonal" than the old ones (the blue).  This is because they are less correlated.  We can check this more quantitatively by looking at the matrix:

# %%
f.parameter_correlation_matrix

# %% [markdown]
# And we see that now the $(F0,F1)$ element is smaller, $0.29$.

# %%
