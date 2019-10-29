"""Classes used for evaluation of prior probabilities

Initially this handles priors on single parameters that don't depend on any
other parameters.  This will need to be supplemented with a mechanism, probably
in the model class, that implements priors on combinations of parameters,
such as total proper motion, 2-d sky location, etc.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats
from astropy import log
from scipy.stats import norm, rv_continuous, rv_discrete, uniform


class Prior(object):
    r"""Class for evaluation of prior probability densities

    Any Prior object returns the probability density using
    the ``pdf()`` and ``logpdf()`` methods.  For generality, these
    are written so that they work on a scalar value or a numpy array
    of values.

    Parameters
    ----------
    _rv : rv_frozen
        Private member that holds an instance of rv_frozen used
        to evaluate the prior. It must be a 'frozen distribution', with all
        location and shape parameters set.
        See <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>

    Priors are evaluated at values corresponding to the num_value of the parameter
    and don't currently use units (the num_unit is the assumed unit)

    Examples
    --------
    A uniform prior of F0, with no bounds (any value is acceptable)

        >>> model.F0.prior = Prior(UniformUnboundedRV())

    A uniform prior on F0 between 50 and 60 Hz (because num_unit is Hz)

        >>> model.F0.prior = Prior(UniformBoundedRV(50.0,60.0))

    A Gaussian prior on PB with mean 32 days and std dev 1.0 day

        >>> model.PB.prior = Prior(scipy.stats.norm(loc=32.0,scale=1.0))

    A bounded gaussian prior that ensure that eccentrity never gets > 1.0

        >>> model.ECC.prior = Prior(GaussianBoundedRV(loc=0.9,scale=0.1,
        ...            lower_bound=0.0,upper_bound=1.0))

    """

    def __init__(self, rv):
        self._rv = rv
        pass

    def pdf(self, value):
        # The astype() calls prevent unsafe cast messages
        if type(value) == np.ndarray:
            v = value.astype(np.float64, casting="same_kind")
        else:
            v = np.float(value)
        return self._rv.pdf(v)

    def logpdf(self, value):
        if type(value) == np.ndarray:
            v = value.astype(np.float64, casting="same_kind")
        else:
            v = np.float(value)
        return self._rv.logpdf(v)


class UniformUnboundedRV(rv_continuous):
    r"""A uniform prior distribution (equivalent to no prior)

    """

    # The astype() calls prevent unsafe cast messages
    def _pdf(self, x):
        return np.ones_like(x).astype(np.float64, casting="same_kind")

    def _logpdf(self, x):
        return np.zeros_like(x).astype(np.float64, casting="same_kind")


def UniformBoundedRV(lower_bound, upper_bound):
    r"""A uniform prior between two bounds

    Parameters
    ----------
    lower_bound : number
        Lower bound of allowed parameter range

    upper_bound : number
        Upper bound of allowed parameter range

    Returns a frozen rv_continuous instance with a uniform probability
    inside the range lower_bound to upper_bound and 0.0 outside
    """
    uu = uniform(lower_bound, (upper_bound - lower_bound))
    return uu


class GaussianRV_gen(rv_continuous):
    r"""A Gaussian prior between two bounds.
    If you just want a gaussian, use scipy.stats.norm
    This version is for generating bounded gaussians

    Parameters
    ----------
    loc : number
        Mode of the gaussian (default=0.0)

    scale : number
        Standard deviation of the gaussian (default=1.0)
    """

    def _pdf(self, x):
        ret = np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)
        return ret


def GaussianBoundedRV(loc=0.0, scale=1.0, lower_bound=-np.inf, upper_bound=np.inf):
    r"""A gaussian prior between two bounds

    Parameters
    ----------
    lower_bound : number
        Lower bound of allowed parameter range

    upper_bound : number
        Upper bound of allowed parameter range

    Returns a frozen rv_continuous instance with a gaussian probability
    inside the range lower_bound to upper_bound and 0.0 outside
    """
    ymin = (lower_bound - loc) / scale
    ymax = (upper_bound - loc) / scale
    n = GaussianRV_gen(name="bounded_gaussian", a=ymin, b=ymax)
    nn = n(loc=loc, scale=scale)
    return nn
