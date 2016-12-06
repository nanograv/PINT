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
    the pdf() and logpdf() methods.  For generality, these 
    are written so that they work on a scalar value or a numpy array
    of values.

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
        model.F0.prior = Prior(UniformRV())

        # A uniform prior on F0 between 50 and 60 Hz (because num_unit is Hz)
        model.F0.prior = Prior(UniformBoundedRV(50.0,60.0))

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

class UniformRV(rv_continuous):
    r"""A uniform prior distribution (equivalent to no prior)

    """

    # The astype() calls prevent unsafe cast messages
    def _pdf(self,x):
        return np.ones_like(x).astype(np.float64,casting='same_kind')
    def _logpdf(self,x):
        return np.zeros_like(x).astype(np.float64,casting='same_kind')

class UniformBoundedRV(rv_continuous):
    r"""A uniform prior between two bounds

    Parameters
    ----------
    lower_bound : number
        Lower bound of allowed parameter range

    upper_bound : number
        Upper bound of allowed parameter range
    """
    def __init__(self,lower_bound,upper_bound):
        super(UniformBoundedRV,self).__init__()
        self.lower = lower_bound
        self.upper = upper_bound
        self.norm = np.float64(1.0/(upper_bound-lower_bound))

    def _pdf(self,x):
        ret = np.where(np.logical_and(x>=self.lower,x<=self.upper),self.norm,0.0)
        return ret

    def _logpdf(self,x):
        ret = np.where(np.logical_and(x>=self.lower,x<=self.upper),
                       np.log(self.norm), -np.inf)
        return ret

class GaussianBoundedRV(rv_continuous):
    r"""A uniform prior between two bounds

    Parameters
    ----------
    lower_bound : number
        Lower bound of allowed parameter range

    upper_bound : number
        Upper bound of allowed parameter range
    """
    def __init__(self,loc,scale,lower_bound,upper_bound):
        super(GaussianBoundedRV,self).__init__()
        self.lower = lower_bound
        self.upper = upper_bound
        self.gaussian = scipy.stats.norm(loc=loc,scale=scale)
        # Norm should be the integral of the gaussian from lower to upper
        # so that integral over allowed range will be == 1.0
        self.norm = 1.0/(self.gaussian.cdf(upper_bound)-self.gaussian.cdf(lower_bound))

    def _pdf(self,x):
        ret = np.where(np.logical_and(x>=self.lower,x<=self.upper), 
            self.norm*self.gaussian.pdf(x), 0.0)
        return ret

    def _logpdf(self,x):
        ret = np.where(np.logical_and(x>=self.lower,x<=self.upper),
                       np.log(self.norm)*self.gaussian.logpdf(x), -np.inf)
        return ret

