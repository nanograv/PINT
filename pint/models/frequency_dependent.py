"""This module implements a frequency evolution of pulsar profiles model"""
from warnings import warn
from . import parameter as p
from .timing_model import TimingModel, MissingParameter
import astropy.units as u
import numpy as np
import pint.utils as ut
import astropy.time as time

class FD(TimingModel):
    """This class provides a timing model for frequency evolution of pulsar
    profiles model.
    """
    register = True
    def __init__(self):
        super(FD, self).__init__()
        self.add_param(p.prefixParameter(name='FD1', units="second", value=0.0,
                       descriptionTplt=lambda x: ("%d term of frequency"
                                                  " dependent  coefficients" % x),
                       unitTplt=lambda x: 'second',
                       type_match='float'))

        self.delay_funcs['L1'] += [self.FD_delay]
        self.order_number = 4
        self.print_par_func = 'print_par_FD'
    def setup(self):
        super(FD, self).setup()
        # Check if FD terms are in order.
        FD_mapping = self.get_prefix_mapping('FD')
        FD_terms = list(FD_mapping.keys())
        FD_terms.sort()
        FD_in_order = list(range(1,max(FD_terms)+1))
        if not FD_terms == FD_in_order:
            diff = list(set(FD_in_order) - set(FD_terms))
            raise MissingParameter("FD", "FD%d"%diff[0])
        self.num_FD_terms = len(FD_terms)
        # set up derivative functions
        for ii, val in FD_mapping.items():
            self.register_deriv_funcs(self.d_delay_FD_d_FDX, 'delay', val)

    def FD_delay(self, toas):
        """This is a function for calculation of frequency dependent delay.
        Z. Arzoumanian, The NANOGrav Nine-year Data Set: Observations, Arrival
        Time Measurements, and Analysis of 37 Millisecond Pulsars, The
        Astrophysical Journal, Volume 813, Issue 1, article id. 65, 31 pp.(2015).
        Eq.(2):
        FDdelay = sum(c_i * (log(obs_freq/1GHz))^i)
        """
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for frequency dependent delay!")
            bfreq = toas['freq']
        FD_mapping = self.get_prefix_mapping('FD')
        log_freq = np.log(bfreq / (1 * u.GHz))
        FD_coeff = [getattr(self, FD_mapping[ii]).value \
                   for ii in range(self.num_FD_terms,0,-1)]
        FD_coeff += [0.0] # Zeroth term of polynomial

        FD_delay = np.polyval(FD_coeff, log_freq)

        return FD_delay * self.FD1.units

    def d_delay_FD_d_FDX(self, toas, param):
        """This is a derivative function for FD parameter
        """
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for frequency dependent delay derivative!")
            bfreq = toas['freq']
        log_freq = np.log(bfreq / (1 * u.GHz))
        FD_par = getattr(self, param)
        FD_term = FD_par.index
        FD_mapping = self.get_prefix_mapping('FD')
        if FD_term > self.num_FD_terms:
            raise ValueError('FD model has no FD%d term' % FD_term)
        # make the selected FD coefficient 1, others 0
        FD_coeff = np.zeros(len(FD_mapping)+1)
        FD_coeff[-1-FD_term] = np.longdouble(1.0)
        d_delay_d_FD = np.polyval(FD_coeff, log_freq)
        return d_delay_d_FD * u.second / FD_par.units

    def make_delay_FD_deriv_funcs(self, param):
        FD_term = getattr(self, param).index
        def deriv_func(toas):
            return self.d_binary_FD_d_FDX(toas, FD_term)
        deriv_func.__name__ = 'd_delay_FD_d_' + param
        setattr(self, 'd_delay_FD_d_' + param, deriv_func)

    def print_par_FD(self):
        result = ''
        FD_mapping = self.get_prefix_mapping('FD')
        for FD in FD_mapping.values():
            FD_par = getattr(self, FD)
            result += FD_par.as_parfile_line()
        return result
