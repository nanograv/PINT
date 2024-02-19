# -*- coding: utf-8 -*-
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
# # Understanding Fitters
#
#

# %%
from IPython.display import display_markdown

import pint.toa
import pint.models
import pint.fitter
import pint.config
import pint.logging

pint.logging.setup(level="INFO")

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support

quantity_support()

# %%
# Load some TOAs and a model to fit
m, t = pint.models.get_model_and_toas(
    pint.config.examplefile("NGC6440E.par"), pint.config.examplefile("NGC6440E.tim")
)

# %%
# You can check if a model includes a noise model with correlated errors (e.g. ECORR or TNRED) by checking the has_correlated_errors property
m.has_correlated_errors

# %% [markdown]
# There are several fitters in PINT, each of which is a subclass of `Fitter`
#
# * `DownhillWLSFitter` - PINT's workhorse fitter, which does a basic weighted least-squares minimization of the residuals.
# * `DownhillGLSFitter` - A generalized least squares fitter, like "tempo -G", that can handle noise processes like ECORR and red noise that are specified by their correlation function properties.
# * `WidebandDownhillFitter` - A fitter that uses DM estimates associated with each TOA. Also supports generalized least squares.
# * `PowellFitter` - A very simple example fitter that uses the Powell method implemented in scipy. One notable feature is that it does not require evaluating derivatives w.r.t the model parameters.
# * `MCMCFitter` - A fitter that does an MCMC fit using the [emcee](https://emcee.readthedocs.io/en/stable/) package. This can be very slow, but accomodates Priors on the parameter values and can produce corner plots and other analyses of the posterior distributions of the parameters.
# * `WLSFitter`, `GLSFitter`, `WidebandFitter` - Simpler fitters that make no attempt to ensure convergence.
#
# You can normally use the function `pint.fitter.Fitter.auto(toas, model)` to construct an appropriate fitter for your model and data.
#

# %% [markdown]
# ## Weighted Least Squares Fitter

# %%
# Instantiate a fitter
wlsfit = pint.fitter.DownhillWLSFitter(toas=t, model=m)

# %% [markdown]
# A fit is performed by calling `fit_toas()`
#
# For most fitters, multiple iterations can be limited by setting the `maxiter` keyword argument.
#
# Downhill fitters will raise the `pint.fitter.MaxiterReached` exception if they stop before detecting convergence; you can capture this exception and continue if you don't mind not having the best-fit answer.

# %%
try:
    wlsfit.fit_toas(maxiter=1)
except pint.fitter.MaxiterReached:
    print("Fitter has not fully converged.")

# %%
# A summary of the fit and resulting model parameters can easily be printed
# Only free parameters will have values and uncertainties in the Postfit column
wlsfit.print_summary()

# %%
# The WLS fitter doesn't handle correlated errors
wlsfit.resids.model.has_correlated_errors

# %%
# You can request a pretty-printed covariance matrix
cov = wlsfit.get_parameter_covariance_matrix(pretty_print=True)

# %%
# plot() will make a plot of the post-fit residuals
wlsfit.plot()

# %% [markdown]
#
# ## Comparing models
#
# There also a convenience function for pretty printing a comparison of two models with the differences measured in sigma.

# %%
display_markdown(wlsfit.model.compare(wlsfit.model_init, format="markdown"), raw=True)

# %% [markdown]
# You can see just how much F1 changed.  Let's compare the $\chi^2$ values:

# %%
print(f"Pre-fit chi-squared value: {wlsfit.resids_init.chi2}")
print(f"Post-fit chi-squared value: {wlsfit.resids.chi2}")

# %% [markdown]
# ## Generalized Least Squares fitter
#
# The GLS fitter is capable of handling correlated noise models.
#
# It has some more complex options using the `maxiter`, `threshold`, and `full_cov` keyword arguments to `fit_toas()`.
#
# If `maxiter` is less than one, **no fitting is done**, just the
# chi-squared computation. In this case, you must provide the `residuals`
# argument.
#
# If `maxiter` is one or more, so fitting is actually done, the
# chi-squared value returned is only approximately the chi-squared
# of the improved(?) model. In fact it is the chi-squared of the
# solution to the linear fitting problem, and the full non-linear
# model should be evaluated and new residuals produced if an accurate
# chi-squared is desired.
#
# A first attempt is made to solve the fitting problem by Cholesky
# decomposition, but if this fails singular value decomposition is
# used instead. In this case singular values below threshold are removed.
#
# `full_cov` determines which calculation is used. If True, the full
# covariance matrix is constructed and the calculation is relatively
# straightforward but the full covariance matrix may be enormous.
# If False, an algorithm is used that takes advantage of the structure
# of the covariance matrix, based on information provided by the noise
# model. The two algorithms should give the same result up to numerical
# accuracy where they both can be applied.

# %% [markdown]
# To test this fitter properly, we need a model that includes correlated noise components, so we will load one from NANOGrav 9yr data release.

# %%
m1855 = pint.models.get_model(pint.config.examplefile("B1855+09_NANOGrav_9yv1.gls.par"))

# %%
# You can check if a model includes a noise model with correlated errors (e.g. ECORR or TNRED) by checking the has_correlated_errors property
m1855.has_correlated_errors

# %%
print(m1855)

# %%
ts1855 = pint.toa.get_TOAs(
    pint.config.examplefile("B1855+09_NANOGrav_9yv1.tim"), model=m1855
)
ts1855.print_summary()

# %% [markdown]
# There is currently a problem with `DownhillGLSFitter`: it doesn't record appropriate noise parameters.

# %%
glsfit = pint.fitter.GLSFitter(toas=ts1855, model=m1855)

# %%
m1855.DMX_0001.prefix

# %%
glsfit.fit_toas(maxiter=1)

# %%
glsfit.print_summary()

# %% [markdown]
# The GLS fitter produces two types of residuals, the normal residuals to the deterministic model and those from the noise model.

# %%
glsfit.resids.time_resids

# %%
glsfit.resids.noise_resids

# %%
# Here we can plot both the residuals to the deterministic model as well as the realization of the noise model residuals
# The difference will be the "whitened" residuals
fig, ax = plt.subplots(figsize=(16, 9))
mjds = glsfit.toas.get_mjds()
ax.plot(mjds, glsfit.resids.time_resids, ".")
ax.plot(mjds, glsfit.resids.noise_resids["pl_red_noise"], ".")

# %% [markdown]
# ## Choosing fitters
#
# You can use the automatic fitter selection to help you choose between `WLSFitter`, `GLSFitter`, and their wideband variants.
# The default `Downhill` fitters generally have better performance than the plain variants.

# %%
autofit = pint.fitter.Fitter.auto(toas=ts1855, model=m1855)

# %%
autofit.fit_toas()

# %%
display_markdown(autofit.model.compare(glsfit.model, format="markdown"), raw=True)

# %% [markdown]
# The results are (thankfully) identical.

# %% [markdown]
# The MCMC fitter is considerably more complicated, so it has its own dedicated walkthroughs in `MCMC_walkthrough.ipynb`
# (for photon data) and `examples/fit_NGC6440E_MCMC.py` (for fitting TOAs).
