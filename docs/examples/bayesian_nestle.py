from pint.models import get_model_and_toas
from pint.bayesian import BayesianTiming
from pint.config import examplefile
from pint.models.priors import Prior
from scipy.stats import uniform

import numpy as np
import nestle
import corner
import sys

# Read the par and tim files
parfile = examplefile("NGC6440E.par.good")
timfile = examplefile("NGC6440E.tim")
model, toas = get_model_and_toas(parfile, timfile)

# Now set the priors.
# I am cheating here by setting these priors around the maximum likelihood estimates.
# In the real world, these priors should be informed by, e.g. previous timing solutions, VLBI localization etc.
# Note that unbounded uniform priors don't work here.
for par in model.free_params:
    param = getattr(model, par)
    param_min = float(param.value - 10 * param.uncertainty_value)
    param_span = float(20 * param.uncertainty_value)
    param.prior = Prior(uniform(param_min, param_span))

# This object provides lnlikelihood, lnprior and prior_transform functions
bt = BayesianTiming(model, toas)


# This order is the same as model.free_params.
print("Free parameters : ", bt.param_labels)


# The default print_progress function in nestle causes too much slowdown.
def print_progress(info):
    if info["it"] % 20 == 0:
        print("\r\033[Kit={:6d} logz={:8f}".format(info["it"], info["logz"]), end="")
        sys.stdout.flush()


# Now run the sampler.
res = nestle.sample(
    bt.lnlikelihood,
    bt.prior_transform,
    bt.nparams,
    method="multi",
    npoints=150,
    callback=print_progress,
)


fig = corner.corner(
    res.samples,
    weights=res.weights,
    labels=bt.param_labels,
    range=[0.9999] * bt.nparams,
)


param_means, param_cov = nestle.mean_and_cov(res.samples, weights=res.weights)
param_stds = np.diag(param_cov) ** 0.5


print("Parameter means and standard deviations")
print("===============================================")
print("Param\t\tMean\t\t\tStd")
print("===============================================")
for par, mean, std in zip(bt.param_labels, param_means, param_stds):
    print(f"{par}\t\t{mean:0.15e}\t{std}")
print("===============================================")
print()

print("===============================================")
np.set_printoptions(precision=2)
print("Parameter Covariance Matrix")
print("===============================================")
print(param_cov)
