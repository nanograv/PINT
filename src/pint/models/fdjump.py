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
    observing frequency/logarithm of observing frequency in the SSB frame.
    This is intended to compensate for the delays introduced by frequency-dependent
    profile structure when a different profiles are used for different systems.

    The default behavior is to have FDJUMPs as polynomials of the observing
    frequency (rather than log-frequency). This is different from the convention
    used for global FD parameters. This choice is made to be compatible with tempo2.
    This is controlled using the FDJUMPLOG parameter. "FDJUMPLOG Y" may not be
    tempo2-compatible.

    Note
    ----
    FDJUMPs have two indices: the polynomial/FD/prefix index and the system/mask
    index. i.e., they have properties of both maskParameters such as JUMPs and
    prefixParameters such as FDs. There is currently no elegant way in PINT to implement
    such parameters due to the way parameter indexing is implemented; there is no way to
    distinguish between mask and prefix indices.

    Hence, they are implemented here as maskParameters as a stopgap measure.
    This means that there must be an upper limit for the FD indices. This is controlled
    using the `pint.models.fdjump.fdjump_max_index` variable, and is 20 by default.
    Note that this is strictly a limitation of the implementation and not a property
    of FDJUMPs themselves.

    FDJUMPs appear in tempo2-format par files as "FDJUMPp", where p is the FD index.
    The mask index is not explicitly mentioned in par files similar to JUMPs.
    PINT understands both "FDJUMPp" and "FDpJUMP" as the same parameter in par files,
    but the internal representation is always "FDpJUMPq", where q is the mask index.

    PINT understands 'q' as the mask parameter just fine, but the identification of 'p'
    as the prefix parameter is done in a hacky way.

    This implementation may be overhauled in the future.

    Parameters supported:

    .. paramtable::
        :class: pint.models.fdjump.FDJump
    """

    register = True
    category = "fdjump"

    def __init__(self):
        super().__init__()

        # Matches "FDpJUMPq" where p and q are integers.
        self.param_regex = re.compile("^FD(\\d+)JUMP(\\d+)")

        self.add_param(
            boolParameter(
                name="FDJUMPLOG",
                value=False,
                description="Whether to use log-frequency (Y) or linear-frequency (N) for computing FDJUMPs.",
            )
        )
        for j in range(1, fdjump_max_index + 1):
            self.add_param(
                maskParameter(
                    name=f"FD{j}JUMP",
                    units="second",
                    description=f"System-dependent FD parameter of polynomial index {j}",
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
        """Extract the FD index from an FDJUMP parameter name. In a parameter name
        "FDpJUMPq", p is the FD/prefix index and q is the mask index.

        Parameters
        ----------
        par: Parameter name (str)

        Returns
        -------
        FD index (int)
        """
        if m := self.param_regex.match(par):
            return int(m.groups()[0])
        else:
            raise ValueError(
                f"The given parameter {par} does not correspond to an FDJUMP."
            )

    def get_freq_y(self, toas):
        """Get frequency or log-frequency in GHz based on the FDJUMPLOG value.
        Returns (freq/1_GHz) if FDJUMPLOG==N and log(freq/1_GHz) if FDJUMPLOG==Y.
        Any non-finite values are replaced by zero.

        Parameters
        ----------
        toas: pint.toa.TOAs

        Returns
        -------
        (freq/1_GHz) or log(freq/1_GHz) depending on the value of FDJUMPLOG (float).
        """
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

        If FDJUMPLOG is Y, use the following expression (similar to global FD parameters):

            FDJUMP_delay = sum_i(c_i * (log(obs_freq/1GHz))^i)

        If FDJUMPLOG is N, use the following expression (same as in tempo2, default):

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
        """Derivative of delay w.r.t. FDJUMP parameters."""
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

    def print_par(self, format="pint"):
        par = super().print_par(format)

        if format != "tempo2":
            return par

        for fdjump in self.fdjumps:
            if getattr(self, fdjump).quantity is not None:
                j = self.get_fd_index(fdjump)
                par = par.replace(f"FD{j}JUMP", f"FDJUMP{j}")

        return par
