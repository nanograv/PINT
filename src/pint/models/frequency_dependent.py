"""A frequency evolution of pulsar profiles model"""
from __future__ import absolute_import, division, print_function

from warnings import warn

import astropy.units as u
import numpy as np

from pint.models.parameter import prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter


class FD(DelayComponent):
    """A timing model for frequency evolution of pulsar profiles."""

    register = True
    category = "frequency_dependent"

    def __init__(self):
        super(FD, self).__init__()
        self.add_param(
            prefixParameter(
                name="FD1",
                units="second",
                value=0.0,
                descriptionTplt=lambda x: (
                    "%d term of frequency" " dependent  coefficients" % x
                ),
                unitTplt=lambda x: "second",
                type_match="float",
            )
        )

        self.delay_funcs_component += [self.FD_delay]

    def setup(self):
        super(FD, self).setup()
        # Check if FD terms are in order.
        FD_mapping = self.get_prefix_mapping_component("FD")
        self.num_FD_terms = len(FD_mapping)
        # set up derivative functions
        for val in FD_mapping.values():
            self.register_deriv_funcs(self.d_delay_FD_d_FDX, val)

    def validate(self):
        super(FD, self).validate()
        FD_terms = list(self.get_prefix_mapping_component("FD").keys())
        FD_terms.sort()
        FD_in_order = list(range(1, max(FD_terms) + 1))
        if not FD_terms == FD_in_order:
            diff = list(set(FD_in_order) - set(FD_terms))
            raise MissingParameter("FD", "FD%d" % diff[0])

    def FD_delay(self, toas, acc_delay=None):
        """Calculate frequency dependent delay.

        Z. Arzoumanian, The NANOGrav Nine-year Data Set: Observations, Arrival
        Time Measurements, and Analysis of 37 Millisecond Pulsars, The
        Astrophysical Journal, Volume 813, Issue 1, article id. 65, 31 pp.(2015).
        Eq.(2):
        FDdelay = sum(c_i * (log(obs_freq/1GHz))^i)
        """
        tbl = toas.table
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for frequency dependent delay!")
            bfreq = tbl["freq"]
        FD_mapping = self.get_prefix_mapping_component("FD")
        log_freq = np.log(bfreq.to_value(u.GHz))
        non_finite = np.invert(np.isfinite(log_freq))
        log_freq[non_finite] = 0.0
        FD_coeff = [
            getattr(self, FD_mapping[ii]).value
            for ii in range(self.num_FD_terms, 0, -1)
        ]
        FD_coeff += [0.0]  # Zeroth term of polynomial

        FD_delay = np.polyval(FD_coeff, log_freq)

        return FD_delay * self.FD1.units

    def d_delay_FD_d_FDX(self, toas, param, acc_delay=None):
        """This is a derivative function for FD parameter
        """
        tbl = toas.table
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn(
                "Using topocentric frequency for frequency dependent delay derivative!"
            )
            bfreq = tbl["freq"]
        log_freq = np.log(bfreq / (1 * u.GHz))
        non_finite = np.invert(np.isfinite(log_freq))
        log_freq[non_finite] = 0.0
        FD_par = getattr(self, param)
        FD_term = FD_par.index
        FD_mapping = self.get_prefix_mapping_component("FD")
        if FD_term > self.num_FD_terms:
            raise ValueError("FD model has no FD%d term" % FD_term)
        # make the selected FD coefficient 1, others 0
        FD_coeff = np.zeros(len(FD_mapping) + 1)
        FD_coeff[-1 - FD_term] = np.longdouble(1.0)
        d_delay_d_FD = np.polyval(FD_coeff, log_freq)
        return d_delay_d_FD * u.second / FD_par.units

    def print_par(self):
        result = ""
        FD_mapping = self.get_prefix_mapping_component("FD")
        for FD in FD_mapping.values():
            FD_par = getattr(self, FD)
            result += FD_par.as_parfile_line()
        return result
