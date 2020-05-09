from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from scipy.linalg import LinAlgError
from astropy import log

from pint.phase import Phase
from pint.utils import weighted_mean

__all__ = ["Residuals"]

# also we import from fitter, down below to avoid circular relative imports


class Residuals(object):
    """Class to compute residuals between TOAs and a TimingModel
    
    Parameters
    ----------
    subtract_mean : bool
        Controls whether mean will be subtracted from the residuals
    use_weighted_mean : bool
        Controls whether mean compution is weighted (by errors) or not.
    track_mode : "nearest", "use_pulse_numbers"
        Controls how pulse numbers are assigned. The default "nearest" assigns each TOA to the nearest integer pulse.
        "use_pulse_numbers" uses the pulse_number column of the TOAs table to assign pulse numbers. This mode is
        selected automatically if the model has parameter TRACK == "-2".
    """

    def __init__(
        self,
        toas=None,
        model=None,
        subtract_mean=True,
        use_weighted_mean=True,
        track_mode="nearest",
    ):
        self.toas = toas
        self.model = model
        self.subtract_mean = subtract_mean
        self.use_weighted_mean = use_weighted_mean
        self.track_mode = track_mode
        if getattr(self.model, "TRACK").value == "-2":
            self.track_mode = "use_pulse_numbers"
        if toas is not None and model is not None:
            self.phase_resids = self.calc_phase_resids()
            self.time_resids = self.calc_time_resids()
            self.dof = self.get_dof()
        else:
            self.phase_resids = None
            self.time_resids = None
        # delay chi-squared computation until needed to avoid infinite recursion
        # also it's expensive
        # only relevant if there are correlated errors
        self._chi2 = None
        self.noise_resids = {}

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

    def rms_weighted(self):
        """Compute weighted RMS of the residals in time."""
        if np.any(self.toas.get_errors() == 0):
            raise ValueError(
                "Some TOA errors are zero - cannot calculate weighted RMS of residuals"
            )
        w = 1.0 / (self.toas.get_errors().to(u.s) ** 2)

        wmean, werr, wsdev = weighted_mean(self.time_resids, w, sdev=True)
        return wsdev.to(u.us)

    def calc_phase_resids(self):
        """Return timing model residuals in pulse phase."""

        # Read any delta_pulse_numbers that are in the TOAs table.
        # These are for PHASE statements, -padd flags, as well as user-inserted phase jumps
        # Check for the column, and if not there then create it as zeros
        try:
            delta_pulse_numbers = Phase(self.toas.table["delta_pulse_number"])
        except:
            self.toas.table["delta_pulse_number"] = np.zeros(len(self.toas.get_mjds()))
            delta_pulse_numbers = Phase(self.toas.table["delta_pulse_number"])

        # Track on pulse numbers, if requested
        if self.track_mode == "use_pulse_numbers":
            pulse_num = self.toas.get_pulse_numbers()
            if pulse_num is None:
                raise ValueError(
                    "Pulse numbers missing from TOAs but track_mode requires them"
                )
            # Compute model phase. For pulse numbers tracking
            # we need absolute phases, since TZRMJD serves as the pulse
            # number reference.
            modelphase = (
                self.model.phase(self.toas, abs_phase=True) + delta_pulse_numbers
            )
            # First assign each TOA to the correct relative pulse number, including
            # and delta_pulse_numbers (from PHASE lines or adding phase jumps in GUI)
            residualphase = modelphase - Phase(pulse_num, np.zeros_like(pulse_num))
            # This converts from a Phase object to a np.float128
            full = residualphase.int + residualphase.frac
        # If not tracking then do the usual nearest pulse number calculation
        else:
            # Compute model phase
            modelphase = self.model.phase(self.toas) + delta_pulse_numbers
            # Here it subtracts the first phase, so making the first TOA be the
            # reference. Not sure this is a good idea.
            if self.subtract_mean:
                modelphase -= Phase(modelphase.int[0], modelphase.frac[0])

            # Here we discard the integer portion of the residual and replace it with 0
            # This is effectively selecting the nearst pulse to compute the residual to.
            residualphase = Phase(np.zeros_like(modelphase.frac), modelphase.frac)
            # This converts from a Phase object to a np.float128
            full = residualphase.int + residualphase.frac
        # If we are using pulse numbers, do we really want to subtract any kind of mean?
        if not self.subtract_mean:
            return full
        if not self.use_weighted_mean:
            mean = full.mean()
        else:
            # Errs for weighted sum.  Units don't matter since they will
            # cancel out in the weighted sum.
            if np.any(self.toas.get_errors() == 0):
                raise ValueError(
                    "Some TOA errors are zero - cannot calculate residuals"
                )
            w = 1.0 / (self.toas.get_errors().value ** 2)
            mean, err = weighted_mean(full, w)
        return full - mean

    def calc_time_resids(self):
        """Return timing model residuals in time (seconds)."""
        if self.phase_resids is None:
            self.phase_resids = self.calc_phase_resids()
        return (self.phase_resids / self.get_PSR_freq()).to(u.s)

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
        # Now subtract 1 for the implicit global offset parameter
        # Note that we should do two things eventually
        # 1. Make the offset not be a hidden parameter
        # 2. Have a model object return the number of free parameters instead of having to count non-frozen parameters like above
        dof -= 1
        return dof

    def get_reduced_chi2(self):
        """Return the weighted reduced chi-squared for the model and toas."""
        return self.calc_chi2() / self.get_dof()

    def update(self):
        """Recalculate everything in residuals class after changing model or TOAs"""
        if self.toas is None or self.model is None:
            self.phase_resids = None
            self.time_resids = None
        if self.toas is None:
            raise ValueError("No TOAs provided for residuals update")
        if self.model is None:
            raise ValueError("No model provided for residuals update")

        self.phase_resids = self.calc_phase_resids()
        self.time_resids = self.calc_time_resids()
        self._chi2 = None  # trigger chi2 recalculation when needed
        self.dof = self.get_dof()

    def ecorr_average(self, use_noise_model=True):
        """
        Uses the ECORR noise model time-binning to compute "epoch-averaged"
        residuals.  Requires ECORR be used in the timing model.  If
        use_noise_model is true, the noise model terms (EFAC, EQUAD, ECORR) will
        be applied to the TOA uncertainties, otherwise only the raw
        uncertainties will be used.

        Returns a dictionary with the following entries:

          mjds           Average MJD for each segment

          freqs          Average topocentric frequency for each segment

          time_resids    Average residual for each segment, time units

          noise_resids   Dictionary of per-noise-component average residual

          errors         Uncertainty on averaged residuals

          indices        List of lists giving the indices of TOAs in the original
                         TOA table for each segment
        """

        # ECORR is required
        try:
            ecorr = self.model.get_components_by_category()["ecorr_noise"][0]
        except KeyError:
            raise ValueError("ECORR not present in noise model")

        # "U" matrix gives the TOA binning, "weight" is ECORR
        # uncertainty in seconds, squared.
        U, ecorr_err2 = ecorr.ecorr_basis_weight_pair(self.toas)
        ecorr_err2 *= u.s * u.s

        if use_noise_model:
            err = self.model.scaled_sigma(self.toas)
        else:
            err = self.toas.get_errors()
            ecorr_err2 *= 0.0

        # Weight for sums, and normalization
        wt = 1.0 / (err * err)
        a_norm = np.dot(U.T, wt)

        def wtsum(x):
            return np.dot(U.T, wt * x) / a_norm

        # Weighted average of various quantities
        avg = {}
        avg["mjds"] = wtsum(self.toas.get_mjds())
        avg["freqs"] = wtsum(self.toas.get_freqs())
        avg["time_resids"] = wtsum(self.time_resids)
        avg["noise_resids"] = {}
        for k in self.noise_resids.keys():
            avg["noise_resids"][k] = wtsum(self.noise_resids[k])

        # Uncertainties
        # TODO could add an option to incorporate residual scatter
        avg["errors"] = np.sqrt(1.0 / a_norm + ecorr_err2)

        # Indices back into original TOA list
        avg["indices"] = [list(np.where(U[:, i])[0]) for i in range(U.shape[1])]

        return avg


class GeneralResiduals(Residuals):
    """ Subclass for generalized residuals.

    This class computes for the residuals from TOAs and other independently
    measured data (e.g., DM values from wide band TOAs). The independently
    measured data have to be at the same epoch as the TOAs (i.e.,
    non_TOA_residuals(TOA) = non_TOA_measurements(TOA) -  non_TOA_model(TOA))

    Note
    ----
    If the non-TOA data is not provided separately, this class will search such
    data in the TOA class. The model for non_TOA data must be a part of the
    timing model (e.g., DM values as function of TOA).
    """

    def __init__(self, toas=None, model=None, non_TOA_data={},
                 non_TOA_model=[], weighted_mean=True,
                 set_pulse_nums=False):
        # Construct the triditional TOA residuals first.
        super(GeneralResiduals, self).__init__(toas=toas,
                                               model=model,
                                               weighted_mean=True,
                                               set_pulse_nums=False)

        # Check the input for non-TOA data and model.
        
