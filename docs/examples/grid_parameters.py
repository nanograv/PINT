# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python [conda env:pintdev]
#     language: python
#     name: conda-env-pintdev-py
# ---

# %% [markdown]
# # Fit data over a grid of parameters
# 1. Load a data-set
# 2. Find the best-fit
# 3. Determine $\chi^2$ of a 1D grid of parameters
# 4. Determine $\chi^2$ over a grid of 2 parameters
# 5. Plot the contours of $\chi^2$ over that grid, and compare the 1D and 2D confidence regions

# %%
import astropy.units as u
import numpy as np

import pint.config
import pint.gridutils
import pint.models.parameter as param
import pint.residuals
from pint.fitter import GLSFitter
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

f = GLSFitter(t, m)
# find the best-fit
f.fit_toas()
bestfit = f.resids.chi2

# %% [markdown]
# We'll do something like 3-sigma around the best-fit values of $F0$ and $F1$

# %%
F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        25,
    )
F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        27,
    )

# %%
# Do a 1D "grid"  Make sure that the parameters are supplied as tuples with length 1
chi2_F0 = pint.gridutils.grid_chisq(f, ("F0",), (F0,))

# %%
# find the uncertainty by looking where Delta chi^2=1
F0_delta_chi2 = np.interp(1,chi2_F0[F0>=f.model.F0.quantity]-chi2_F0.min(),F0[F0>=f.model.F0.quantity].astype(np.float64))

# %%
fig, ax = plt.subplots(figsize=(16,9))
# just plot the values offset from the best-fit values
ax.plot(F0 - f.model.F0.quantity, chi2_F0,label='$\chi^2$ curve')
ax.plot(0, chi2_F0.min(), 'ro',markersize=12,label='Minimum $\chi^2$')
ax.plot((f.model.F0.uncertainty)*np.ones(2),[59,69],'r:',label='Uncertainty')
ax.plot((-f.model.F0.uncertainty)*np.ones(2),[59,69],'r:')
ax.plot([F0_delta_chi2-f.model.F0.quantity,F0_delta_chi2-f.model.F0.quantity,-(F0_delta_chi2-f.model.F0.quantity),-(F0_delta_chi2-f.model.F0.quantity)],
       [chi2_F0.min(),chi2_F0.min()+1,chi2_F0.min()+1,chi2_F0.min()],'g--',label='$\Delta \chi^2=1$')
#ax.errorbar(0,0,xerr=f.model.F0.uncertainty,yerr=f.model.F1.uncertainty,fmt='ro')
ax.set_xlabel('$\Delta F_0$ (Hz)',fontsize=24)
ax.set_ylabel('$\chi^2$',fontsize=24)
ax.set_ylim([59,69])
ax.legend()

# %% [markdown]
# From the above you can see that the calculated uncertainties on $F0$ from the model fit agree with where $\Delta \chi^2=1$, which is what we would expect for a single parameter.

# %% [markdown]
# Now, compute a 2D grid of $\chi^2(F0,F1)$

# %% tags=[]
chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1))

# %% [markdown]
# We want to plot contour levels appropriate for the joint confidence contours with 2 parameters.  This is discussed many places (e.g., https://ned.ipac.caltech.edu/level5/Wall2/Wal3_4.html).  Rather than look up the values in a table we will compute them ourselves.
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
nsigma = np.arange(1,4)
# these are the CDFs going from -infinity to nsigma.  So subtract away 0.5 and double for the 2-sided values
CIs = (scipy.stats.norm().cdf(nsigma)-0.5)*2
print(f"Confidence intervals for {nsigma} sigma: {CIs}")
# chi^2 random variable for 2 parameters
rv = scipy.stats.chi2(2)
x = np.linspace(0,20)
contour_levels = np.interp(CIs, rv.cdf(x), x)
print(f"Contour levels for {nsigma} sigma and 2 parameters: {contour_levels}")

# %% [markdown]
# Let's repeat that for a single parameter for comparison

# %%
# 1, 2, and 3 sigma confidence limits
nsigma = np.arange(1,4)
# these are the CDFs going from -infinity to nsigma.  So subtract away 0.5 and double for the 2-sided values
CIs = (scipy.stats.norm().cdf(nsigma)-0.5)*2
print(f"Confidence intervals for {nsigma} sigma: {CIs}")
# chi^2 random variable for 2 parameters
rv = scipy.stats.chi2(1)
x = np.linspace(0,20)
contour_levels_1param = np.interp(CIs, rv.cdf(x), x)
print(f"Contour levels for {nsigma} sigma and 1 parameter: {contour_levels_1param}")

# %%
fig, ax = plt.subplots(figsize=(16,9))
# just plot the values offset from the best-fit values
ax.contour(F0-f.model.F0.quantity,F1-f.model.F1.quantity,chi2grid-bestfit,levels=contour_levels,colors="b")
ax.contour(F0-f.model.F0.quantity,F1-f.model.F1.quantity,chi2grid-bestfit,levels=contour_levels_1param,colors="g",linestyles='--')
ax.errorbar(0,0,xerr=f.model.F0.uncertainty,yerr=f.model.F1.uncertainty,fmt='ro')
ax.set_xlabel('$\Delta F_0$ (Hz)',fontsize=24)
ax.set_ylabel('$\Delta F_1$ (Hz/s)',fontsize=24)

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
