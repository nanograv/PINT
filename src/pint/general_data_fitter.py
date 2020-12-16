from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy

import astropy.units as u
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
from astropy import log
import astropy.constants as const
import pint.utils
from pint.fitter import Fitter


class GRSFitter(Fitter):
    """ Generalized residual fitter.

    The generalized residual fitter uses more than TOAs as data input, it also
    uses the independent measured data (e.g., Wideband's DM valus) as part of
    the fit data.

    Note
    ----
    This fitter assume the independent measured data are stored in the toas as
    default.
    """

    def __init__(self, toas, model, residuals=None):
        super(GRSFitter, self).__init__(toas=toas, model=model, residuals=residuals)
        self.method = "g"
