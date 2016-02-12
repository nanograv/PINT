"""priors.py
Defines classes used for evaluation prior probabilities

Initially this handles priors on single parameters that don't depend on any
other parameters.  This will need to be supplemented with a mechanism, probably
in the model class, that implements priors on combinations of parameters,
such as total proper motion, 2-d sky location, etc.

"""
from __future__ import division, absolute_import, print_function
import scipy.stats
from scipy.stats import rv_continuous, rv_discrete, norm

from astropy import log
import numpy as np

class Prior(object):
    r"""Class for evaluation of prior probability densities
    
     Any Prior object returns the probability density using
    the prior_probability() method.  For generality, these 
    are written so that they work on a scalar value or a numpy array
    of values.
    
    The default Prior is just the constant value 1.0.
    Subclasses implement other forms.
   
    """
    
    def __init__(self, rv):
        """
        _rv : rv_frozen
            Private member that holds an instance of rv_frozen used
            to evaluate the prior. It must be a 'frozen distribution', with all
            location and shape parameter set.
            See <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>

        Priors are evaluated at values corresponding to the num_value of the parameter
        and don't currently use units (the num_unit is the assumed unit)
        
        Examples
        --------
        # A uniform prior of F0
        model.F0.prior = Prior(UniformPrior())
        
        # A uniform prior on F0 between 50 and 60 Hz (because num_unit is Hz)
        model.F0.prior = Prior(UniformBoundedPrior(50.0,60.0))
        
        # A Gaussian prior on PB with mean 32 days and std dev 1.0 day
        model.PB.prior = Prior(scipy.stats.norm(loc=32.0,scale=1.0))
        

        """
        self._rv = rv
        pass
    
    def pdf(self,value):
        # The astype() calls prevent unsafe cast messages
        if type(value) == np.ndarray:
            v = value.astype(np.float64,casting='same_kind')
        else:
            v = np.float(value)
        return self._rv.pdf(v)
        
    def logpdf(self,value):
        if type(value) == np.ndarray:
            v = value.astype(np.float64,casting='same_kind')
        else:
            v = np.float(value)
        return self._rv.logpdf(v)
        
class UniformPrior(rv_continuous):
    r"""A uniform prior distribution (equivalent to no prior)
    
    """
    
    # The astype() calls prevent unsafe cast messages
    def _pdf(self,x):
        log.info('x ({0}) {1}'.format(type(x),x))
        return np.ones_like(x).astype(np.float64,casting='same_kind')
    def _logpdf(self,x):
        return np.zeros_like(x).astype(np.float64,casting='same_kind')

class UniformBoundedPrior(rv_continuous):
    r"""A uniform prior between two bounds
    
    Parameters
    ----------
    lower_bound : number
        Lower bound of allowed parameter range
        
    upper_bound : number
        Upper bound of allowed parameter range
    """
    def __init__(self,lower_bound,upper_bound):
        super(UniformBoundedPrior,self).__init__()
        self.lower = lower_bound
        self.upper = upper_bound
        self.norm = np.float64(1.0/(upper_bound-lower_bound))
                
    def _pdf(self,x):
        ret = np.where(np.logical_and(x>self.lower,x<self.upper),self.norm,0.0)
        return ret
        
    def _logpdf(self,x):
        ret = np.where(np.logical_and(x>self.lower,x<self.upper),
                       np.log(self.norm), -np.inf)
        return ret
       
if __name__ == '__main__':
    val1 = 2.0
    vals = np.linspace(0.0,10,11.0)
    prior1 = Prior(UniformPrior())
    print(val1,prior1.pdf(val1))
    print(vals,prior1.logpdf(vals))
    
    prior2 = Prior(norm(1.0,3.0))
    print(val1,prior2.pdf(val1))
    print(vals,prior2.logpdf(vals))

    prior3 = Prior(UniformBoundedPrior(3.0,6.0))
