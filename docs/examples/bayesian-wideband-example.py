# %% [markdown]
# # PINT Bayesian Interface Example (Wideband)

# %%
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from pint.bayesian import BayesianTiming
from pint.config import examplefile
from pint.fitter import WidebandDownhillFitter
from pint.logging import setup as setup_log
from pint.models import get_model_and_toas

# %%
# Turn off log messages. They can slow down the processing.
setup_log(level="WARNING")

# %%
# This is a simulated dataset.
m, t = get_model_and_toas(examplefile("test-wb-0.par"), examplefile("test-wb-0.tim"))

# %%
# Fit the model to the data to get the parameter uncertainties.
ftr = WidebandDownhillFitter(t, m)
ftr.fit_toas()
m = ftr.model

# %%
# Now set the priors.
# I am cheating here by setting the priors around the maximum likelihood estimates.
# This is a bad idea for real datasets and can bias the estimates. I am doing this
# here just to make everything finish faster. In the real world, these priors should
# be informed by, e.g. previous (independent) timing solutions, pulsar search results,
# VLBI localization etc. Note that unbounded uniform priors don't work here.
prior_info = {}
for par in m.free_params:
    param = getattr(m, par)
    param_min = float(param.value - 10 * param.uncertainty_value)
    param_max = float(param.value + 10 * param.uncertainty_value)
    prior_info[par] = {"distr": "uniform", "pmin": param_min, "pmax": param_max}

# %%
# Set the EFAC and DMEFAC priors and unfreeze them.
# Don't do this before the fitting step. The fitter doesn't know
# how to deal with noise parameters.
prior_info["EFAC1"] = {"distr": "normal", "mu": 1, "sigma": 0.1}
prior_info["DMEFAC1"] = {"distr": "normal", "mu": 1, "sigma": 0.1}

m.EFAC1.frozen = False
m.EFAC1.uncertainty_value = 0.01
m.DMEFAC1.frozen = False
m.DMEFAC1.uncertainty_value = 0.01

# %%
# The likelihood function behaves better if `use_pulse_numbers==True`.
bt = BayesianTiming(m, t, use_pulse_numbers=True, prior_info=prior_info)

# %%
print("Number of parameters = ", bt.nparams)
print("Likelihood method = ", bt.likelihood_method)

# %%
nwalkers = 25
sampler = emcee.EnsembleSampler(nwalkers, bt.nparams, bt.lnposterior)

# %%
# Start the sampler close to the maximul likelihood estimate.
maxlike_params = np.array([param.value for param in bt.params], dtype=float)
maxlike_errors = [param.uncertainty_value for param in bt.params]
start_points = (
    np.repeat([maxlike_params], nwalkers).reshape(bt.nparams, nwalkers).T
    + np.random.randn(nwalkers, bt.nparams) * maxlike_errors
)

# %%
# ** IMPORTANT!!! **
# This is used to exclude the following time-consuming steps from the readthedocs build.
# Set this to False while actually using this example.
rtd = True

# %%
if not rtd:
    print("Running emcee...")
    chain_length = 1000
    sampler.run_mcmc(
        start_points,
        chain_length,
        progress=True,
    )

    samples_emcee = sampler.get_chain(flat=True, discard=100)

# %%
# Plot the chains to make sure they have converged and the burn-in has been removed properly.
if not rtd:
    for idx, param_chain in enumerate(samples_emcee.T):
        plt.subplot(bt.nparams, 1, idx + 1)
        plt.plot(param_chain)
        plt.ylabel(bt.param_labels[idx])
        plt.autoscale()
    plt.show()

# %%
if not rtd:
    fig = corner.corner(
        samples_emcee, labels=bt.param_labels, quantiles=[0.5], truths=maxlike_params
    )
    plt.show()
