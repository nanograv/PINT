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
# # PINT Bayesian Interface Examples

# %%
from pint.models import get_model, get_model_and_toas
from pint.bayesian import BayesianTiming
from pint.config import examplefile
from pint.models.priors import Prior
from pint.logging import setup as setup_log
from scipy.stats import uniform

# %%
import numpy as np
import emcee
import nestle
import corner
import io
import matplotlib.pyplot as plt

# %%
# Turn off log messages. They can slow down the processing.
setup_log(level="WARNING")

# %%
# Read the par and tim files
parfile = examplefile("NGC6440E.par.good")
timfile = examplefile("NGC6440E.tim")
model, toas = get_model_and_toas(parfile, timfile)

# %%
# This is optional, but the likelihood function behaves better if
# we have the pulse numbers. Make sure that your timing solution is
# phase connected before doing this.
toas.compute_pulse_numbers(model)

# %%
# Now set the priors.
# I am cheating here by setting the priors around the maximum likelihood estimates.
# This is a bad idea for real datasets and can bias the estimates. I am doing this
# here just to make everything finish faster. In the real world, these priors should
# be informed by, e.g. previous (independent) timing solutions, pulsar search results,
# VLBI localization etc. Note that unbounded uniform priors don't work here.
for par in model.free_params:
    param = getattr(model, par)
    param_min = float(param.value - 10 * param.uncertainty_value)
    param_span = float(20 * param.uncertainty_value)
    param.prior = Prior(uniform(param_min, param_span))

# %%
# Now let us create a BayesianTiming object. This is a wrapper around the
# PINT API that provides provides lnlikelihood, lnprior and prior_transform
# functions which can be passed to a sampler of your choice.
bt = BayesianTiming(model, toas, use_pulse_numbers=True)

# %%
print("Number of parameters = ", bt.nparams)
print("Likelihood method = ", bt.likelihood_method)

# %% [markdown]
# ## MCMC sampling using emcee

# %%
nwalkers = 20
sampler = emcee.EnsembleSampler(nwalkers, bt.nparams, bt.lnposterior)

# %%
# Choose the MCMC start points in the vicinity of the maximum likelihood estimate
# available in the `model` object. This helps the MCMC chains converge faster.
# We can also draw these points from the prior, but the MCMC chains will converge
# slower in that case.
maxlike_params = np.array([param.value for param in bt.params], dtype=float)
maxlike_errors = [param.uncertainty_value for param in bt.params]
start_points = (
    np.repeat([maxlike_params], nwalkers).reshape(bt.nparams, nwalkers).T
    + np.random.randn(nwalkers, bt.nparams) * maxlike_errors
)

# %%
# ** IMPORTANT!!! **
# This is used to exclude some of the following time-consuming steps from the readthedocs build.
# Set this to False while actually using this example.
rtd = True

# %%
# Use longer chain_length for real runs. It is kept small here so that
# the sampling finishes quickly (and because I know the burn in is short
# because of the cheating priors above).
if not rtd:
    print("Running emcee...")

    chain_length = 1000

    sampler.run_mcmc(
        start_points,
        chain_length,
        progress=True,
    )

# %%
if not rtd:
    # Merge all the chains together after discarding the first 100 samples as 'burn-in'.
    # The burn-in should be decided after looking at the chains in the real world.
    samples_emcee = sampler.get_chain(flat=True, discard=100)

    # Plot the MCMC chains to make sure that the burn-in has been removed properly.
    # Otherwise, go back and discard more points.
    for idx, param_chain in enumerate(samples_emcee.T):
        plt.subplot(bt.nparams, 1, idx + 1)
        plt.plot(param_chain, label=bt.param_labels[idx])
        plt.legend()
    plt.show()

# %%
# Plot the posterior distribution.
if not rtd:
    fig = corner.corner(samples_emcee, labels=bt.param_labels)
    plt.show()

# %% [markdown]
# ## Nested sampling with nestle

# %% [markdown]
# Nested sampling computes the Bayesian evidence along with posterior samples.
# This allows us to do compare two models. Let us compare the model above with
# and without an EFAC.

# %%
# Let us run the model without EFAC first. We can reuse the `bt` object from before.

# Nesle is really simple :)
# method='multi' runs the MultiNest algorithm.
# `npoints` is the number of live points.
# `dlogz` is the target accuracy in the computed Bayesian evidence.
# Increasing `npoints` or decreasing `dlogz` gives more accurate results,
# but at the cost of time.
if not rtd:
    print("Running nestle...")
    result_nestle_1 = nestle.sample(
        bt.lnlikelihood,
        bt.prior_transform,
        bt.nparams,
        method="multi",
        npoints=150,
        dlogz=0.5,
        callback=nestle.print_progress,
    )

# %%
# Plot the posterior
# The nested samples come with weights, which must be taken into account
# while plotting.
if not rtd:
    fig = corner.corner(
        result_nestle_1.samples,
        weights=result_nestle_1.weights,
        labels=bt.param_labels,
        range=[0.999] * bt.nparams,
    )
    plt.show()

# %% [markdown]
# Let us create a new model with an EFAC applied to all toas (all
# TOAs in this dataset are from GBT).

# %%
# casting the model to str gives the par file representation.
# Add an EFAC to the par file and make it unfrozen.
parfile = f"{str(model)}EFAC TEL gbt 1 1"
model2 = get_model(io.StringIO(parfile))

# %%
# Now set the priors.
# Again, don't do this with real data. Use uninformative priors or priors
# motivated by previous experiments. This is done here with the sole purpose
# of making the run finish fast. Let us try this with the prior_info option now.
prior_info = {}
for par in model2.free_params:
    param = getattr(model2, par)
    param_min = float(param.value - 10 * param.uncertainty_value)
    param_max = float(param.value + 10 * param.uncertainty_value)
    prior_info[par] = {"distr": "uniform", "pmin": param_min, "pmax": param_max}

prior_info["EFAC1"] = {"distr": "normal", "mu": 1, "sigma": 0.1}

# %%
bt2 = BayesianTiming(model2, toas, use_pulse_numbers=True, prior_info=prior_info)
print(bt2.likelihood_method)

# %%
if not rtd:
    result_nestle_2 = nestle.sample(
        bt2.lnlikelihood,
        bt2.prior_transform,
        bt2.nparams,
        method="multi",
        npoints=150,
        dlogz=0.5,
        callback=nestle.print_progress,
    )

# %%
# Plot the posterior.
# The EFAC looks consistent with 1.
if not rtd:
    fig2 = corner.corner(
        result_nestle_2.samples,
        weights=result_nestle_2.weights,
        labels=bt2.param_labels,
        range=[0.999] * bt2.nparams,
    )
    plt.show()

# %% [markdown]
# Now let us look at the evidences and compute the Bayes factor.

# %%
if not rtd:
    print(
        f"Evidence without EFAC : {result_nestle_1.logz} +/- {result_nestle_1.logzerr}"
    )
    print(f"Evidence with EFAC : {result_nestle_2.logz} +/- {result_nestle_2.logzerr}")

    bf = np.exp(result_nestle_1.logz - result_nestle_2.logz)
    print(f"Bayes factor : {bf} (in favor of no EFAC)")

# %% [markdown]
# The Bayes factor tells us that the EFAC is unnecessary for this dataset.
