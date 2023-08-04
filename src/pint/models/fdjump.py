"""System and frequency dependent delays to model profile evolution."""

import re
from warnings import warn

import astropy.units as u
import numpy as np

from pint.models.parameter import boolParameter, maskParameter
from pint.models.timing_model import DelayComponent

fdjump_max_index = 10


class FDJump(DelayComponent):
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
        if m := self.param_regex.match(par):
            return int(m.groups()[0])
        else:
            raise ValueError(
                f"The given parameter {par} does not correspond to an FDJUMP."
            )

    def get_freq_y(self, toas):
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
        y = self.get_freq_y(toas)

        delay = np.zeros_like(y)
        for fdjump in self.fdjumps:
            mask = fdjump.select_toa_mask(toas)
            ymask = y[mask]
            fdidx = self.get_fd_index(fdjump)
            fdcoeff = getattr(self, fdjump).value
            delay[mask] += fdcoeff * ymask**fdidx

        return delay * u.s

    def d_delay_d_FDJUMP(self, toas, param, acc_delay=None):
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
