"""priors.py
Defines classes used for evaluation prior probabilities

Initially this handles priors on single parameters that don't depend on any
other parameters.  This will need to be supplemented with a mechanism, probably
in the model class, that implements priors on combinations of parameters,
such as total proper motion, 2-d sky location, etc.

"""
from __future__ import division, absolute_import, print_function

import numpy as np

class Prior(object):
    
    def __init__(self):
        pass
    
    def prior_probability(self,value):
        return np.ones_like(value)
        

class UniformBoundedPrior(Prior):
    def __init__(self,lower_bound,upper_bound):
        super(UniformBoundedPrior,self).__init__()
        self.lower = lower_bound
        self.upper = upper_bound
                
    def prior_probability(self,value):
        return np.logical_and(value>self.lower,value<self.upper)*np.ones_like(value)

class GaussianPrior(Prior):
    def __init__(self, mean, sigma):
        super(GaussianPrior,self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.var = sigma**2

    def prior_probability(self,value):
        return np.exp(-(value-self.mean)/(2*self.var))/np.sqrt(2.0*np.pi*self.var)
       
if __name__ == '__main__':
    val1 = 2.0
    vals = np.linspace(0.0,10,11.0)
    #prior1 = Prior()
    #print(val1,prior1.prior_probability(val1))
    #print(vals,prior1.prior_probability(vals))
    prior2 = GaussianPrior(1.0,3.0)
    print(val1,prior2.prior_probability(val1))
    print(vals,prior2.prior_probability(vals))

