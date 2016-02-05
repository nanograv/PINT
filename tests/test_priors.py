#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import pint.models
from pint.models.priors import *

m = pint.models.get_model('examples/NGC6440E.par')

print("F0 is {0}".format(m.F0.num_value))
# Check with uniform prior
print("Uniform Prior on F0 {0}".format(m.F0.prior_probability()))

# Now add a gaussian prior with mean 61.0 and sigma 2.0
m.F0.prior = GaussianPrior(61.0,2.0)
print("Gaussian Prior on F0 {0}".format(m.F0.prior_probability()))

# Now test a value before setting
print("Test Gaussian Prior of F0 {0}".format(m.F0.prior_probability(value=63.0)))

