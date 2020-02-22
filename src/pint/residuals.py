from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from scipy.linalg import LinAlgError
from astropy import log

from pint import dimensionless_cycles
from pint.phase import Phase

__all__ = ["Residuals"]

# also we import from fitter, down below to avoid circular relative imports


class Residuals(object):
    """Residual(toa=None, model=None)"""

    def __init__(self, toas=None, model=None, weighted_mean=True, set_pulse_nums=False):
        self.toas = toas
        self.model = model
        if toas is not None and model is not None:
            self.phase_resids = self.calc_phase_resids(
                weighted_mean=weighted_mean, set_pulse_nums=set_pulse_nums
            )
            self.time_resids = self.calc_time_resids(weighted_mean=weighted_mean)
            self.dof = self.get_dof()
        else:
            self.phase_resids = None
            self.time_resids = None
        # delay chi-squared computation until needed to avoid infinite recursion
        # also it's expensive
        # only relevant if there are correlated errors
        self._chi2 = None

    @property
    def chi2_reduced(self):
        return self.chi2 / self.dof

    @property
    def chi2(self):
        """Compute chi-squared as needed and cache the result"""
        if self._chi2 is None:
            self._chi2 = self.calc_chi2()
        assert self._chi2 is not None
        return self._chi2

    def calc_phase_resids(self, weighted_mean=True, set_pulse_nums=False):
        """Return timing model residuals in pulse phase."""
        # Please define what set_pulse_nums means!

        # Read any delta_pulse_numbers that are in the TOAs table.
        # These are for PHASE statements as well as user-inserted phase jumps
        # Check for the column, and if not there then create it as zeros
        try:
            delta_pulse_numbers = Phase(self.toas.table["delta_pulse_number"])
        except:
            self.toas.table["delta_pulse_number"] = np.zeros(len(self.toas.get_mjds()))
            delta_pulse_numbers = Phase(self.toas.table["delta_pulse_number"])

        # I have no idea what this is trying to do. It just sets delta_pulse_number to zero
        if set_pulse_nums:
            self.toas.table["delta_pulse_number"] = np.zeros(len(self.toas.get_mjds()))
            delta_pulse_numbers = Phase(self.toas.table["delta_pulse_number"])

        # Compute model phase
        rs = self.model.phase(self.toas)

        # Track on pulse numbers, if requested
        if getattr(self.model, "TRACK").value == "-2":
            pulse_num = self.toas.get_pulse_numbers()
            if pulse_num is None:
                raise ValueError(
                    "Pulse numbers missing from TOAs but TRACK -2 requires them"
                )
            # Compute model phase. For pulse numbers tracking
            # we need absolute phases, since TZRMJD serves as the pulse
            # number reference.
            rs = self.model.phase(self.toas, abs_phase=True)
            # First assign each TOA to the correct relative pulse number
            rs -= Phase(pulse_num, np.zeros_like(pulse_num))
            # Then subtract the constant offset since that is irrelevant
            rs -= Phase(rs.int[0], rs.frac[0])
            full = rs + delta_pulse_numbers
            full = full.int + full.frac

        # If not tracking then do the usual nearest pulse number calculation
        else:
            # Compute model phase
            rs = self.model.phase(self.toas)
            # Here it subtracts the first phase, so making the first TOA be the
            # reference. Not sure this is a good idea.
            rs -= Phase(rs.int[0], rs.frac[0])

            # What exactly is full?
            full = Phase(np.zeros_like(rs.frac), rs.frac) + delta_pulse_numbers
            # This converts full from a Phase object to a np.float128
            full = full.int + full.frac

        # If we are using pulse numbers, do we really want to subtract any kind of mean?
        # Perhaps there should be an option to not subtract any mean?
        if not weighted_mean:
            full -= full.mean()
        else:
            # Errs for weighted sum.  Units don't matter since they will
            # cancel out in the weighted sum.
            if np.any(self.toas.get_errors() == 0):
                raise ValueError(
                    "Some TOA errors are zero - cannot calculate residuals"
                )
            w = 1.0 / (np.array(self.toas.get_errors()) ** 2)
            wm = (full * w).sum() / w.sum()
            full -= wm
        return full

    def calc_time_resids(self, weighted_mean=True):
        """Return timing model residuals in time (seconds)."""
        if self.phase_resids is None:
            self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean)
        with u.set_enabled_equivalencies(dimensionless_cycles):
            return (self.phase_resids.to(u.Unit("")) / self.get_PSR_freq()).to(u.s)

    def get_PSR_freq(self, modelF0=True):
        if modelF0:
            """Return pulsar rotational frequency in Hz. model.F0 must be defined."""
            if self.model.F0.units != "Hz":
                ValueError("F0 units must be Hz")
            # All residuals require the model pulsar frequency to be defined
            F0names = ["F0", "nu"]  # recognized parameter names, needs to be changed
            nF0 = 0
            for n in F0names:
                if n in self.model.params:
                    F0 = getattr(self.model, n).value
                    nF0 += 1
            if nF0 == 0:
                raise ValueError(
                    "no PSR frequency parameter found; "
                    + "valid names are %s" % F0names
                )
            if nF0 > 1:
                raise ValueError(
                    "more than one PSR frequency parameter found; "
                    + "should be only one from %s" % F0names
                )
            return F0 * u.Hz
        return self.model.d_phase_d_toa(self.toas)

    def calc_chi2(self, full_cov=False):
        """Return the weighted chi-squared for the model and toas.

        If the errors on the TOAs are independent this is a straightforward
        calculation, but if the noise model introduces correlated errors then
        obtaining a meaningful chi-squared value requires a Cholesky
        decomposition. This is carried out, here, by constructing a GLSFitter
        and asking it to do the chi-squared computation but not a fit.

        The return value here is available as self.chi2, which will not
        redo the computation unless necessary.

        The chi-squared value calculated here is suitable for use in downhill
        minimization algorithms and Bayesian approaches.

        Handling of problematic results - degenerate conditions explored by
        a minimizer for example - may need to be checked to confirm that they
        correctly return infinity.
        """
        if self.model.has_correlated_errors:
            # Use GLS but don't actually fit
            from pint.fitter import GLSFitter

            f = GLSFitter(self.toas, self.model, residuals=self)
            try:
                return f.fit_toas(maxiter=0, full_cov=full_cov)
            except LinAlgError as e:
                log.warning(
                    "Degenerate conditions encountered when "
                    "computing chi-squared: %s" % (e,)
                )
                return np.inf
        else:
            # Residual units are in seconds. Error units are in microseconds.
            if (self.toas.get_errors() == 0.0).any():
                return np.inf
            else:
                # The self.time_resids is in the unit of "s", the error "us".
                # This is more correct way, but it is the slowest.
                # return (((self.time_resids / self.toas.get_errors()).decompose()**2.0).sum()).value

                # This method is faster then the method above but not the most correct way
                # return ((self.time_resids.to(u.s) / self.toas.get_errors().to(u.s)).value**2.0).sum()

                # This the fastest way, but highly depend on the assumption of time_resids and
                # error units.
                return (
                    (self.time_resids / self.toas.get_errors().to(u.s)) ** 2.0
                ).sum()

    def get_dof(self):
        """Return number of degrees of freedom for the model."""
        dof = self.toas.ntoas
        for p in self.model.params:
            dof -= bool(not getattr(self.model, p).frozen)
        return dof

    def get_reduced_chi2(self):
        """Return the weighted reduced chi-squared for the model and toas."""
        return self.calc_chi2() / self.get_dof()

    def update(self, weighted_mean=True):
        """Recalculate everything in residuals class after changing model or TOAs"""
        if self.toas is None or self.model is None:
            self.phase_resids = None
            self.time_resids = None
        if self.toas is None:
            raise ValueError("No TOAs provided for residuals update")
        if self.model is None:
            raise ValueError("No model provided for residuals update")

        self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean)
        self.time_resids = self.calc_time_resids(weighted_mean=weighted_mean)
        self._chi2 = None  # trigger chi2 recalculation when needed
        self.dof = self.get_dof()
