"""A wideband TOAs fitter (This fitter name may change or be integrated to the
base fitter class)"""
from __future__ import absolute_import, print_function, division
import copy, numbers
import numpy as np
import astropy.units as u
import abc
import scipy.optimize as opt, scipy.linalg as sl
from .residuals import resids
from .fitter import GLSFitter

class WidebandFitter(GLSFitter):
    """The Wideband Fitter is designed to perform general least fitting with
       noise model and wideband TOAs.

       Parameter
       ---------
       toas : `~pint.toa.TOAs` object
           Input TOAs for fitting.
       model : `~pint.models.TimingModel` object
           Initial timing model.

       Note
       ----
       This fitter class is designed for temporary use of wideband TOAs. Its API
       will change and some functionality may be merged to the base fitter.
    """
    def __init__(self, toas=None, model=None):
        super(WidebandFitter, self).__init__(toas=toas, model=model, )
        self.method = 'Wideband_TOA_fitter'
        self.

    def get_data(self):
        # Get residuals and TOA uncertainties in seconds
        self.update_resids()
        residuals = self.resids.time_resids.to(u.s).value

    def get_data_error(self):


    def get_designmatrix(self, ):
        # Get the base designmatrix
        M, params, units = self.model.designmatrix(toas=self.toas,
                                                   incfrozen=False,
                                                   incoffset=True)
