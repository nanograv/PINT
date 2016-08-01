"""This module implements a frequency evolution of pulsar profiles model"""
from warnings import warn
import parameter as p
from .timing_model import TimingModel, Cache
import astropy.units as u
import numpy as np
import pint.utils as ut
import astropy.time as time

class FD(TimingModel):
    """This class provides a timing model for frequency evolution of pulsar
    profiles model.
    """
    def __init__(self):
        super(FD, self).__init__()
        self.add_param(p.prefixParameter(name='FD1', units="second", value=0.0,
                       descriptionTplt=lambda x: ("%d term of frequency"
                                                  " dependent  coefficients" % x),
                       unitTplt=lambda x: 'second',
                       type_match='float'))

        self.delay_funcs['L1'] += [self.FD_delay]

    def setup(self):
        super(FD, self).setup()
        # Check if FD terms are in order.
        FD_mapping = self.get_prefix_mapping('FD')
        FD_terms = FD_mapping.keys()
        FD_terms.sort()
        FD_in_order = range(1,max(FD_terms)+1)
        if not FD_terms == FD_in_order:
            diff = list(set(FD_in_order) - set(FD_terms))
            raise MissingParameter("FD", "FD%d"%diff[0])
        self.num_FD_terms = len(FD_terms)
        # set up derivative functions
        for ii, val in FD_mapping.iteritems():
            self.make_delay_FD_deriv_funcs(val)
            self.delay_derivs += [getattr(self, 'd_delay_FD_d_' + val)]

    def FD_delay(self, toas):
        """This is a function for calculation of frequency dependent delay.
        Z. Arzoumanian, The NANOGrav Nine-year Data Set: Observations, Arrival
        Time Measurements, and Analysis of 37 Millisecond Pulsars, The
        Astrophysical Journal, Volume 813, Issue 1, article id. 65, 31 pp.(2015).
        Eq.(2):
        FDdelay = sum(c_i * (log(obs_freq/1GHz))^i)
        """
        FD_mapping = self.get_prefix_mapping('FD')
        log_freq = np.log(toas['freq'] / (1 * u.GHz))
        FD_coeff = [getattr(self, FD_mapping[ii]).value \
                   for ii in range(self.num_FD_terms,0,-1)]
        FD_coeff += [0.0]

        FD_delay = np.polyval(FD_coeff, log_freq)

        return FD_delay * self.FD1.units

    def d_delay_FD_d_FDX(self, toas, FD_term=1):
        """This is a derivative function for FD parameter
        """
        FD_mapping = self.get_prefix_mapping('FD')
        if FD_term > self.num_FD_terms:
            raise ValueError('FD model has no FD%d term' % FD_term)

        d_delay_d_FD = np.zeros_like(toas, dtype=np.longdouble)
        for ii in range(1,self.num_FD_terms+1):
            FD_coef = getattr(self, FD_mapping[ii])
            if ii == FD_term:
                FD_coef.value = 1.0
            log_freq = np.log(toas['freq'] / (1 * u.GHz))
            d_delay_d_FD += FD_coef.value * (log_freq) ** ii
        return d_delay_d_FD

    def make_delay_FD_deriv_funcs(self, param):
        FD_term = getattr(self, param).index
        def deriv_func(toas):
            return self.d_binary_FD_d_FDX(toas, FD_term)
        deriv_func.__name__ = 'd_delay_FD_d_' + param
        setattr(self, 'd_delay_FD_d_' + param, deriv_func)
