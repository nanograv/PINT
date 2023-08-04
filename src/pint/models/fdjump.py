"""System and frequency dependent delays to model profile evolution."""

import re
from warnings import warn

import astropy.units as u
import numpy as np

from pint.models.parameter import boolParameter, maskParameter
from pint.models.timing_model import DelayComponent

fdjump_max_index = 20


class FDJump(DelayComponent):
    """A timing model for system-dependent frequency evolution of pulsar
    profiles.

    This model expresses the delay as a polynomial function of the
    observing frequency/logarithm of observing frequency. This is
    intended to compensate for the delays introduced by frequency-dependent
    profile structure when a frequency-independent template profile is used.

    Parameters supported:

    .. paramtable::
        :class: pint.models.fdjump.FDJump
    """

    register = True
    category = "fdjump"

    def __init__(self):
        super().__init__()

        self.param_regex = re.compile("^FD(\\d+)JUMP(\\d+)")

        self.add_param(
            boolParameter(
                name="FDJUMPLOG",
                value=False,
                description="Whether to use log-frequency for computing FDJUMPs.",
            )
        )
        for j in range(1, fdjump_max_index + 1):
            self.add_param(
                maskParameter(
                    name=f"FD{j}JUMP",
                    units="second",
                    description=f"System-dependent FD parameter of index {j}",
                )
            )

        self.delay_funcs_component += [self.fdjump_delay]

    def setup(self):
        super().setup()

        self.fdjumps = [
            mask_par
            for mask_par in self.get_params_of_type("maskParameter")
            if self.param_regex.match(mask_par)
        ]

        for fdj in self.fdjumps:
            # prevents duplicates from being added to phase_deriv_funcs
            if fdj in self.deriv_funcs.keys():
                del self.deriv_funcs[fdj]
            self.register_deriv_funcs(self.d_delay_d_FDJUMP, fdj)

    def get_fd_index(self, par):
        """Extract the index from an FDJUMP parameter name."""
        if m := self.param_regex.match(par):
            return int(m.groups()[0])
        else:
            raise ValueError(
                f"The given parameter {par} does not correspond to an FDJUMP."
            )

    def get_freq_y(self, toas):
        """Get frequency or log-frequency in GHz based on the FDJUMPLOG value."""
        tbl = toas.table
        try:
            freq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for frequency dependent delay!")
            freq = tbl["freq"]

        y = (
            np.log(freq.to(u.GHz).value)
            if self.FDJUMPLOG.value
            else freq.to(u.GHz).value
        )
        non_finite = np.invert(np.isfinite(y))
        y[non_finite] = 0.0

        return y

    def fdjump_delay(self, toas, acc_delay=None):
        """Calculate frequency dependent delay.

        If FDJUMPLOG is Y, use the following expression:
            Z. Arzoumanian, The NANOGrav Nine-year Data Set: Observations, Arrival
            Time Measurements, and Analysis of 37 Millisecond Pulsars, The
            Astrophysical Journal, Volume 813, Issue 1, article id. 65, 31 pp.(2015).
            Eq.(2):
            FDJUMP_delay = sum_i(c_i * (log(obs_freq/1GHz))^i)

        If FDJUMPLOG is N, use the following expression (same as in tempo2):
            FDJUMP_delay = sum_i(c_i * (obs_freq/1GHz)^i)
        """
        y = self.get_freq_y(toas)

        delay = np.zeros_like(y)
        for fdjump in self.fdjumps:
            fdj = getattr(self, fdjump)
            if fdj.quantity is not None:
                mask = fdj.select_toa_mask(toas)
                ymask = y[mask]
                fdidx = self.get_fd_index(fdjump)
                fdcoeff = fdj.value
                delay[mask] += fdcoeff * ymask**fdidx

        return delay * u.s

    def d_delay_d_FDJUMP(self, toas, param, acc_delay=None):
        """Derivative of delay wrt for FDJUMP parameters."""
        assert (
            bool(self.param_regex.match(param))
            and hasattr(self, param)
            and getattr(self, param).quantity is not None
        ), f"{param} is not present in the FDJUMP model."

        y = self.get_freq_y(toas)
        mask = getattr(self, param).select_toa_mask(toas)
        ymask = y[mask]
        fdidx = self.get_fd_index(param)

        delay_derivative = np.zeros_like(y)
        delay_derivative[mask] = ymask**fdidx

        return delay_derivative * u.dimensionless_unscaled
