"""Frequency-dependent delays to model profile evolution."""
from warnings import warn

import astropy.units as u
import numpy as np

from pint.models.parameter import prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter


class FD(DelayComponent):
    """A timing model for frequency evolution of pulsar profiles.

    This model expresses the delay as a polynomial function of the
    logarithm of observing frequency. This is intended to compensate
    for the delays introduced by frequency-dependent profile structure
    when a frequency-independent template profile is used.

    Parameters supported:

    .. paramtable::
        :class: pint.models.frequency_dependent.FD
    """

    @classmethod
    def _description_template(cls, x):
        return f"{x} term of frequency dependent coefficients"

    register = True
    category = "frequency_dependent"

    def __init__(self):
        super().__init__()
        self.add_param(
            prefixParameter(
                name="FD1",
                units="second",
                value=0.0,
                description="Polynomial coefficient of log-frequency-dependent delay",
                # descriptionTplt=lambda x: (
                #    "%d term of frequency" " dependent  coefficients" % x
                # ),
                descriptionTplt=self._description_template,
                # unitTplt=lambda x: "second",
                type_match="float",
            )
        )

        self.delay_funcs_component += [self.FD_delay]

    def setup(self):
        super().setup()
        # Check if FD terms are in order.
        FD_mapping = self.get_prefix_mapping_component("FD")
        self.num_FD_terms = len(FD_mapping)
        # set up derivative functions
        for val in FD_mapping.values():
            self.register_deriv_funcs(self.d_delay_FD_d_FDX, val)

    def validate(self):
        super().validate()
        FD_terms = sorted(self.get_prefix_mapping_component("FD").keys())
        FD_in_order = list(range(1, max(FD_terms) + 1))
        if FD_terms != FD_in_order:
            diff = list(set(FD_in_order) - set(FD_terms))
            raise MissingParameter("FD", "FD%d" % diff[0])

    def FD_delay(self, toas, acc_delay=None):
        """Calculate frequency dependent delay.

        Z. Arzoumanian, The NANOGrav Nine-year Data Set: Observations, Arrival
        Time Measurements, and Analysis of 37 Millisecond Pulsars, The
        Astrophysical Journal, Volume 813, Issue 1, article id. 65, 31 pp.(2015).
        Eq.(2):
            FD_delay = sum_i(c_i * (log(obs_freq/1GHz))^i)
        """
        tbl = toas.table
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for frequency dependent delay!")
            bfreq = tbl["freq"]
        return self.FD_delay_frequency(bfreq)

    def FD_delay_frequency(self, freq):
        """Compute the FD delay at an array of frequencies."""
        FD_mapping = self.get_prefix_mapping_component("FD")
        log_freq = np.log(freq.to(u.GHz).value)
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
        """This is a derivative function for FD parameter"""
        tbl = toas.table
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
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

    def print_par(self, format="pint"):
        result = ""
        FD_mapping = self.get_prefix_mapping_component("FD")
        for FD in FD_mapping.values():
            FD_par = getattr(self, FD)
            result += FD_par.as_parfile_line(format=format)
        return result
