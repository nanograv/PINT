from pint.models import get_model_and_toas
from pint.bayesian import BayesianTiming
from pint.models.priors import Prior
from pint.config import examplefile
from scipy.stats import uniform

import numpy as np
import matplotlib.pyplot as plt
import nestle
import corner

# Wideband par and tim files
parfile = examplefile("J1614-2230_NANOGrav_12yv3.wb.gls.par")
timfile = examplefile("J1614-2230_NANOGrav_12yv3.wb.tim")
model, toas = get_model_and_toas(parfile, timfile)

# The par files have a lot of free parameters.
# Let us keep only a few of them free so that the sampling finishes in a reasonable time.
for param_label, param in model.get_params_dict().items():
    getattr(model, param_label).frozen = True

model.ELAT.frozen = False
model.ELONG.frozen = False
for par in model.free_params:
    param = getattr(model, par)
    param_min = float(param.value - 3 * param.uncertainty_value)
    param_span = float(6 * param.uncertainty_value)
    param.prior = Prior(uniform(param_min, param_span))

model.DM.frozen = False
dm = float(model.DM.value)
model.DM.prior = Prior(uniform(dm - 0.05, dm + 0.05))

model.EFAC1.frozen = False
model.EFAC1.prior = Prior(uniform(0.1, 2))

model.DMEFAC1.frozen = False
model.DMEFAC1.prior = Prior(uniform(0.1, 5))

# Make sure the free parameters are what we expect.
print(model.free_params)

# Create the BayesianTiming object
bt = BayesianTiming(model, toas)

# Test out if the likelihood function works.
test_cube = 0.5 * np.ones(bt.nparams)
test_params = bt.prior_transform(test_cube)
test_lnl = bt.lnlikelihood(test_params)

# Do the sampling using nestle
result_nestle = nestle.sample(
    bt.lnlikelihood,
    bt.prior_transform,
    bt.nparams,
    method="multi",
    npoints=200,
    dlogz=0.5,
    callback=nestle.print_progress,
)

# Visualize the posterior distribution
fig = corner.corner(
    result_nestle.samples,
    weights=result_nestle.weights,
    labels=bt.param_labels,
    range=[0.999] * bt.nparams,
)
plt.show()
