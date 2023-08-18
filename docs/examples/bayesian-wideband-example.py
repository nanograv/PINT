import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from pint.bayesian import BayesianTiming
from pint.config import examplefile
from pint.fitter import WidebandDownhillFitter
from pint.logging import setup as setup_log
from pint.models import get_model_and_toas

setup_log(level="WARNING")

m, t = get_model_and_toas(examplefile("test-wb-0.par"), examplefile("test-wb-0.tim"))

ftr = WidebandDownhillFitter(t, m)
ftr.fit_toas()
m = ftr.model

prior_info = {}
for par in m.free_params:
    param = getattr(m, par)
    param_min = float(param.value - 10 * param.uncertainty_value)
    param_max = float(param.value + 10 * param.uncertainty_value)
    prior_info[par] = {"distr": "uniform", "pmin": param_min, "pmax": param_max}

prior_info["EFAC1"] = {"distr": "normal", "mu": 1, "sigma": 0.1}
prior_info["DMEFAC1"] = {"distr": "normal", "mu": 1, "sigma": 0.1}

m.EFAC1.frozen = False
m.EFAC1.uncertainty_value = 0.01
m.DMEFAC1.frozen = False
m.DMEFAC1.uncertainty_value = 0.01

bt = BayesianTiming(m, t, use_pulse_numbers=True, prior_info=prior_info)

print("Number of parameters = ", bt.nparams)
print("Likelihood method = ", bt.likelihood_method)

nwalkers = 25
sampler = emcee.EnsembleSampler(nwalkers, bt.nparams, bt.lnposterior)

maxlike_params = np.array([param.value for param in bt.params], dtype=float)
maxlike_errors = [param.uncertainty_value for param in bt.params]
start_points = (
    np.repeat([maxlike_params], nwalkers).reshape(bt.nparams, nwalkers).T
    + np.random.randn(nwalkers, bt.nparams) * maxlike_errors
)

print("Running emcee...")
chain_length = 1000
sampler.run_mcmc(
    start_points,
    chain_length,
    progress=True,
)

samples_emcee = sampler.get_chain(flat=True, discard=100)

for idx, param_chain in enumerate(samples_emcee.T):
    plt.subplot(bt.nparams, 1, idx + 1)
    plt.plot(param_chain)
    plt.ylabel(bt.param_labels[idx])
    plt.autoscale()
plt.show()

fig = corner.corner(
    samples_emcee, labels=bt.param_labels, quantiles=[0.5], truths=maxlike_params
)
plt.show()
