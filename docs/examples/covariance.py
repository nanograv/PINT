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
#     display_name: .env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Accessing correlation matrices and model derivatives
#
# The results of a fit consist of best-fit parameter values and uncertainties, and residuals; there are conventional data products from pulsar timing. Additional information can be useful though: we can describe the correlations between model parameters in a matrix, and we can compute the derivatives of the residuals with respect to the model parameters. Both these additional pieces of information can be obtained from a Fitter object in PINT; this notebook will demonstrate how to do this efficiently.
#
# import contextlib
# import os
#
# import astropy.units as u
# %matplotlib inline
# import matplotlib.pyplot as plt
# %%
import numpy as np
import scipy.linalg
import scipy.stats

# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support

import pint.fitter
import pint.models
import pint.toa

pint.logging.setup(level="INFO")
quantity_support()

# %%
parfile = pint.config.examplefile("NGC6440E.par")
timfile = pint.config.examplefile("NGC6440E.tim")
assert os.path.exists(parfile)
assert os.path.exists(timfile)

# %%
m, t = pint.models.get_model_and_toas(parfile, timfile)

# %% [markdown]
# ## Extracting the parameter covariance matrix
#
# Unfortunately, parameter correlation matrices are not stored when `.par` files are recorded, only the individual parameter uncertainties. In PINT, the machinery for computing these matrices resides in `Fitter` objects. We will therefore construct one and carry out a fit - but we will take zero steps, so that the fit doesn't change the solution (and it runs fairly quickly!).
#
# Normally you should probably actually do something if the model isn't converged! Specifically, the covariance matrix probably isn't very useful if you're not at a best-fit set of parameters. Nevertheless, if you're confident your solution is close enough, you can just ignore the fitter's convergence testing by capturing the exception.

# %%
fo = pint.fitter.Fitter.auto(t, m)

with contextlib.suppress(pint.fitter.MaxiterReached):
    fo.fit_toas(maxiter=0)
    print("Actually converged!")

# %% [markdown]
# You can get a human-readable version of the parameter correlation matrix:

# %%
fo.parameter_correlation_matrix

# %% [markdown]
# If you want a machine-readable version:

# %%
fo.parameter_correlation_matrix.labels

# %%
fo.parameter_correlation_matrix.matrix

# %%
fo.parameter_correlation_matrix.labels

# %% [markdown]
# Be warned: if the model includes red noise parameters, there may be more rows and columns than labels in the parameter covariance matrix. These unlabelled rows and columns will always be at the end. Let's check that there aren't surprises waiting for this pulsar.
#
# Even if there are no noise component entries, the correlation matrix includes a row and column for the non-parameter `Offset`. This arises because internally PINT fits allow for a constant offset in phase, but custom in pulsar timing is to report mean-subtracted residuals and ignore the absolute phase.

# %%
print(f"Model free paramters: {len(fo.model.free_params)}")
print(f"Correlation matrix labels: {len(fo.parameter_correlation_matrix.labels[0])}")
print(f"Correlation matrix shape: {fo.parameter_correlation_matrix.shape}")

# %% [markdown]
# Let's summarize the worst covariances.

# %%
correlation = fo.parameter_correlation_matrix.matrix[1:, 1:]
params = fo.model.free_params

correlations = []
for i, p1 in enumerate(params):
    for j, p2 in enumerate(params[:i]):
        correlations.append((p1, p2, correlation[i, j]))

correlations.sort(key=lambda t: -abs(t[-1]))
for p1, p2, c in correlations:
    if abs(c) < 0.5:
        break
    print(f"{p1:10s} {p2:10s} {c:+.15f}")


# %% [markdown]
# ## Error ellipses
#
# In the frequentist least-squares fitting we do in PINT, the model is assumed to be linear over the range of plausible values, and as a result the estimate of the plausible parameter distribution is a multivariate normal distribution (with correlations as computed above). The confidence regions we obtain are therefore ellipsoids. An n-dimensional ellipsoid is rather cumbersome to visualize, but we can find it useful to plot two-dimensional projections. These are analogous to Bayesian posterior distributions and credible regions.
#
# Let's plot the credible region for the pair of parameters `DM` anf `F1`.

# %%
p1 = "F1"
p2 = "DM"
i = params.index(p1)
j = params.index(p2)
cor = np.array([[1, correlation[i, j]], [correlation[i, j], 1]])
sigmas = np.array([fo.get_fitparams_uncertainty()[p] for p in [p1, p2]])
vals, vecs = scipy.linalg.eigh(cor)

for n_sigma in [1, 2, 3]:
    thresh = scipy.stats.chi2(2).isf(2 * scipy.stats.norm.cdf(-n_sigma))
    angles = np.linspace(0, 2 * np.pi, 200)
    points = thresh * (
        vals[0] * np.cos(angles)[:, None] * vecs[None, :, 0]
        + vals[1] * np.sin(angles)[:, None] * vecs[None, :, 1]
    )
    plt.plot(
        points[:, 0] * sigmas[0], points[:, 1] * sigmas[1], label=f"{n_sigma} sigma"
    )

# plt.axhspan(-sigmas[0], sigmas[0], alpha=0.5)
# plt.axvspan(-sigmas[1], sigmas[1], alpha=0.5)

plt.legend()
plt.xlabel(r"$\Delta$" + f"{p1}")
plt.ylabel(r"$\Delta$" + f"{p2}")

# %% [markdown]
# You can generate something like a posterior sample fairly easily:

# %%
all_sigmas = np.array([fo.get_fitparams_uncertainty()[p] for p in params])
sample = (
    scipy.stats.multivariate_normal(cov=correlation).rvs(size=1000)
    * all_sigmas[None, :]
)

# %% [markdown]
# ## Model derivatives
#
# PINT's fitters rely on having analytical derivatives of the timing model with respect to each parameter. These can be obtained by querying appropriate methods in the `TimingModel` object, but it is more conveniently packaged as the "design matrix" for the fit.

# %%
design, names, units = fo.get_designmatrix()
print(names)
print(units)
design.shape
