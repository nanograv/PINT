"""Objects for comparing models to data.

These objects can be constructed directly, as ``Residuals(toas, model)``, or
they are constructed during fitting operations with :class:`pint.fitter.Fitter`
objects, as ``fitter.residual``. Variants exist for arrival-time-only data
(:class:`pint.residuals.Residuals`) and for arrival times that come paired with
dispersion measures (:class:`pint.residuals.WidebandTOAResiduals`).
"""
import collections
import copy
import warnings

import astropy.units as u
import numpy as np
from scipy.linalg import LinAlgError
from loguru import logger as log

from pint.models.dispersion_model import Dispersion
from pint.models.parameter import maskParameter
from pint.phase import Phase
from pint.utils import (
    sherman_morrison_dot,
    weighted_mean,
    taylor_horner_deriv,
    woodbury_dot,
)

__all__ = [
    "Residuals",
    "WidebandTOAResiduals",
    "WidebandDMResiduals",
    "CombinedResiduals",
]


class Residuals:
    """Class to compute residuals between TOAs and a TimingModel.

    This class serves to store the results of a comparison between TOAs and a
    model. It also implements certain basic statistical calculations. This
    class also serves as a base class providing some infrastructure to support
    residuals from other kinds of data/model comparison.

    This class provides access to the residuals in both phase (turns) and time
    (seconds) form through the ``.phase_resids`` and the ``.time_resids``
    attributes.

    Uncertainties on these residuals are available in time units using
    ``.get_data_error()``; this can include or not include any rescaling
    of the uncertainties implied by the model's EFAC or EQUAD.

    Attributes
    ----------
    phase_resids : :class:`astropy.units.Quantity`
        Residuals in phase units
    time_resids : :class:`astropy.units.Quantity`
        Residuals in time units

    Parameters
    ----------
    toas: :class:`pint.toa.TOAs`, optional
        The input TOAs object. Default: None
    model: :class:`pint.models.timing_model.TimingModel`, optional
        Input model object. Default: None
    residual_type: str, optional
        The type of the residuals. Default: 'toa'
    unit: :class:`astropy.units.Unit`, optional
        The default unit of the residuals. Default: u.s
    subtract_mean : bool
        Controls whether mean will be subtracted from the residuals.
        This option will be ignored if a `PhaseOffset` is present in the timing model.
    use_weighted_mean : bool
        Controls whether mean computation is weighted (by errors) or not.
    track_mode : None, "nearest", "use_pulse_numbers"
        Controls how pulse numbers are assigned. ``"nearest"`` assigns
        each TOA to the nearest integer pulse. ``"use_pulse_numbers"`` uses the
        ``pulse_number`` column of the TOAs table to assign pulse numbers. If the
        default, None, is passed, use the pulse numbers if the model has the
        parameter TRACK == "-2" and not if it has TRACK == "0". If neither of the
        above is set, use pulse numbers if there are pulse numbers present and not
        if there aren't.
    """

    def __new__(
        cls,
        toas=None,
        model=None,
        residual_type="toa",
        unit=u.s,
        subtract_mean=True,
        use_weighted_mean=True,
        track_mode=None,
        use_abs_phase=True,
    ):
        if cls is Residuals:
            try:
                cls = residual_map[residual_type.lower()]
            except KeyError as e:
                raise ValueError(
                    f"'{residual_type}' is not a PINT supported residual. Currently supported data types are {list(residual_map.keys())}"
                ) from e

        return super().__new__(cls)

    def __init__(
        self,
        toas=None,
        model=None,
        residual_type="toa",
        unit=u.s,
        subtract_mean=True,
        use_weighted_mean=True,
        track_mode=None,
        use_abs_phase=True,
    ):
        self.toas = toas
        self.model = model
        self.residual_type = residual_type

        if "PhaseOffset" in model.components and subtract_mean:
            log.debug(
                "Disabling implicit `subtract_mean` because `PhaseOffset` is present in the timing model."
            )
        self.subtract_mean = subtract_mean and "PhaseOffset" not in model.components

        self.use_abs_phase = use_abs_phase

        self.use_weighted_mean = use_weighted_mean
        if track_mode is None:
            if getattr(self.model, "TRACK").value == "-2":
                self.track_mode = "use_pulse_numbers"
            elif getattr(self.model, "TRACK").value == "0":
                self.track_mode = "nearest"
            elif "pulse_number" in self.toas.table.columns:
                if np.any(np.isnan(toas.table["pulse_number"])):
                    log.warning(
                        "Some TOAs are missing pulse numbers, they will not be used."
                    )
                    self.track_mode = "nearest"
                else:
                    self.track_mode = "use_pulse_numbers"
            else:
                self.track_mode = "nearest"
        else:
            self.track_mode = track_mode
        if toas is not None and model is not None:
            self.phase_resids = self.calc_phase_resids()
            self.time_resids = self.calc_time_resids()
        else:
            self.phase_resids = None
            self.time_resids = None

        # delay chi-squared computation until needed to avoid infinite recursion
        # also it's expensive
        # only relevant if there are correlated errors
        self._chi2 = None
        self.noise_resids = {}
        # For residual debugging
        self.debug_info = {}
        # We should be carefully for the other type of residuals
        self.unit = unit
        # A flag to identify if this residual object is combined with residual
        # class.
        self._is_combined = False

    @property
    def resids(self):
        """Residuals in time units."""
        if self.time_resids is None:
            self.update()
        return self.time_resids

    @property
    def resids_value(self):
        """Residuals in seconds, with the units stripped."""
        return self.resids.to_value(self.unit)

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

    @property
    def chi2(self):
        """Compute chi-squared as needed and cache the result."""
        if self._chi2 is None:
            self._chi2 = self.calc_chi2()
        assert self._chi2 is not None
        return self._chi2

    @property
    def dof(self):
        """Return number of degrees of freedom for the model."""
        if self._is_combined:
            raise AttributeError(
                "Please use the `.dof` in the CombinedResidual"
                " class. The individual residual's dof is not "
                "calculated correctly in the combined residuals."
            )
        dof = self.toas.ntoas
        # Now subtract 1 for the implicit global offset parameter
        # Note that we should do two things eventually
        # 1. Make the offset not be a hidden parameter
        dof -= len(self.model.free_params) + 1
        return dof

    @property
    def reduced_chi2(self):
        """Return the weighted reduced chi-squared for the model and toas."""
        return self.chi2 / self.dof

    @property
    def chi2_reduced(self):
        """Reduced chi-squared."""
        warnings.warn(
            "Do not use 'residuals.chi2_reduced'. Please use 'residuals.reduced_chi2' instead.",
            DeprecationWarning,
        )

        return self.chi2 / self.dof

    def get_data_error(self, scaled=True):
        """Get errors on time residuals.

        This returns the uncertainties on the time residuals, optionally scaled
        by the noise model.

        Parameters
        ----------
        scaled: bool, optional
            If errors get scaled by the noise model.
        """
        return (
            self.model.scaled_toa_uncertainty(self.toas)
            if scaled
            else self.toas.get_errors()
        )

    def rms_weighted(self):
        """Compute weighted RMS of the residuals in time."""
        # Use scaled errors, if the noise model is not presented, it will
        # return the raw errors
        scaled_errors = self.get_data_error()
        if np.any(scaled_errors.value == 0):
            raise ValueError(
                "Some TOA errors are zero - cannot calculate weighted RMS of residuals"
            )
        w = 1.0 / (scaled_errors.to(u.s) ** 2)

        wmean, werr, wsdev = weighted_mean(self.time_resids, w, sdev=True)
        return wsdev.to(u.us)

    def get_PSR_freq(self, calctype="modelF0"):
        """Return pulsar rotational frequency in Hz.

        Parameters
        ----------
        calctype : {'modelF0', 'numerical', 'taylor'}
            Type of calculation.  If `calctype` == "modelF0", then simply the ``F0``
            parameter from the model.
            If `calctype` == "numerical", then try a numerical derivative
            If `calctype` == "taylor", evaluate the frequency with a Taylor series

        Returns
        -------
        freq : astropy.units.Quantity
            Either the single ``F0`` in the model or the spin frequency at the moment of each TOA.
        """
        assert calctype.lower() in ["modelf0", "taylor", "numerical"]
        if calctype.lower() == "modelf0":
            # TODO this function will be re-write and move to timing model soon.
            # The following is a temporary patch.
            if "Spindown" in self.model.components:
                F0 = self.model.F0.quantity
            elif "P0" in self.model.params:
                F0 = 1.0 / self.model.P0.quantity
            else:
                raise AttributeError(
                    "No pulsar spin parameter(e.g., 'F0'," " 'P0') found."
                )
            return F0.to(u.Hz)
        elif calctype.lower() == "taylor":
            # see Spindown.spindown_phase
            dt = self.model.get_dt(self.toas, 0)
            # if the model is defined through F0, F1, ...
            if "F0" in self.model.params:
                fterms = [0.0 * u.dimensionless_unscaled] + self.model.get_spin_terms()

            # otherwise assume P0, PDOT
            else:
                F0 = 1.0 / self.model.P0.quantity
                if "PDOT" in self.model.params:
                    F1 = -self.model.PDOT.quantity / self.model.P0.quantity**2
                else:
                    F1 = 0 * u.Hz / u.s
                fterms = [0.0 * u.dimensionless_unscaled, F0, F1]
            return taylor_horner_deriv(dt, fterms, deriv_order=1).to(u.Hz)
        elif calctype.lower() == "numerical":
            return self.model.d_phase_d_toa(self.toas)

    def calc_phase_resids(
        self, subtract_mean=None, use_weighted_mean=None, use_abs_phase=None
    ):
        """Compute timing model residuals in pulse phase.

        if ``subtract_mean`` or ``use_weighted_mean`` is None, will use the values set for the object itself

        Parameters
        ----------
        subtract_mean : bool or None, optional
            Subtract the mean of the residuals. This is ignored if the `PhaseOffset` component
            is present in the model. Default is to use the class attribute.
        use_weighted_mean : bool or None, optional
            Whether to use weighted mean for mean subtraction. Default is to use the class attribute.
        use_abs_phase : bool or None, optional
            Whether to use absolute phase (w.r.t. the TZR TOA). Default is to use the class attribute.

        Returns
        -------
        Phase

        See Also
        --------
        :meth:`pint.residuals.Residuals.get_PSR_freq`
        :meth:`pint.residuals.Residuals.calc_time_resids`
        :meth:`pint.residuals.Residuals.calc_whitened_resids`
        """

        if subtract_mean is None:
            subtract_mean = self.subtract_mean

        if "PhaseOffset" in self.model.components and subtract_mean:
            log.debug(
                "Ignoring `subtract_mean` because `PhaseOffset` is present in the timing model."
            )
        subtract_mean = subtract_mean and "PhaseOffset" not in self.model.components

        if use_weighted_mean is None:
            use_weighted_mean = self.use_weighted_mean

        if use_abs_phase is None:
            use_abs_phase = self.use_abs_phase

        # Read any delta_pulse_numbers that are in the TOAs table.
        # These are for PHASE statements, -padd flags, as well as user-inserted phase jumps
        # Check for the column, and if not there then create it as zeros
        if "delta_pulse_number" not in self.toas.table.colnames:
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
                self.model.phase(self.toas, abs_phase=use_abs_phase)
                + delta_pulse_numbers
            )
            # First assign each TOA to the correct relative pulse number, including
            # and delta_pulse_numbers (from PHASE lines or adding phase jumps in GUI)
            i = pulse_num.copy()
            f = np.zeros_like(pulse_num)

            c = np.isnan(pulse_num)
            if np.any(c):
                # i[c] = 0
                raise ValueError("Pulse numbers are missing on some TOAs")
            residualphase = modelphase - Phase(i, f)
            # This converts from a Phase object to a np.float128
            full = residualphase.int + residualphase.frac

        elif self.track_mode == "nearest":
            # Compute model phase
            modelphase = self.model.phase(self.toas) + delta_pulse_numbers
            # Here it subtracts the first phase, so making the first TOA be the
            # reference. Not sure this is a good idea.
            if subtract_mean:
                modelphase -= Phase(modelphase.int[0], modelphase.frac[0])

            # Here we discard the integer portion of the residual and replace it with 0
            # This is effectively selecting the nearest pulse to compute the residual to.
            residualphase = Phase(np.zeros_like(modelphase.frac), modelphase.frac)
            # This converts from a Phase object to a np.float128
            full = residualphase.int + residualphase.frac
        else:
            raise ValueError(f"Invalid track_mode '{self.track_mode}'")

        # If we are using pulse numbers, do we really want to subtract any kind of mean?
        if not subtract_mean:
            return full
        if not use_weighted_mean:
            mean = full.mean()
        else:
            # Errs for weighted sum.  Units don't matter since they will
            # cancel out in the weighted sum.
            if np.any(self.get_data_error() == 0):
                raise ValueError(
                    "Some TOA errors are zero - cannot calculate residuals"
                )
            w = 1.0 / (self.get_data_error().value ** 2)
            mean, err = weighted_mean(full, w)

        return full - mean

    def _calc_mean(self, weighted, type, calctype=None):
        assert type in ["time", "phase"]

        r = (
            self.calc_phase_resids(subtract_mean=False)
            if type == "phase"
            else self.calc_time_resids(subtract_mean=False, calctype=calctype)
        )

        if not weighted:
            return r.mean()

        if np.any(self.get_data_error() == 0):
            raise ValueError("Some TOA errors are zero - cannot calculate residuals")

        w = 1.0 / (self.get_data_error().value ** 2)
        mean, _ = weighted_mean(r, w)
        return mean

    def calc_phase_mean(self, weighted=True):
        """Calculate mean phase of residuals, optionally weighted

        Parameters
        ----------
        weighted : bool, optional

        Returns
        -------
        astropy.units.Quantity
        """
        return self._calc_mean(weighted, "phase")

    def calc_time_mean(self, calctype="taylor", weighted=True):
        """Calculate mean time of residuals, optionally weighted

        Parameters
        ----------
        calctype : str, optional
            Calculation time for phase to time conversion.  See :meth:`pint.residuals.Residuals.calc_time_resids` for details.
        weighted : bool, optional

        Returns
        -------
        astropy.units.Quantity
        """
        return self._calc_mean(weighted, "time", calctype=calctype)

    def calc_time_resids(
        self,
        calctype="taylor",
        subtract_mean=None,
        use_weighted_mean=None,
        use_abs_phase=None,
    ):
        """Compute timing model residuals in time (seconds).

        Converts from phase residuals to time residuals using several possible ways
        to calculate the frequency.

        If ``subtract_mean`` or ``use_weighted_mean`` is None, will use the values set for the object itself

        Parameters
        ----------
        calctype : {'taylor', 'modelF0', 'numerical'}
            Type of calculation.  If `calctype` == "modelF0", then simply the ``F0``
            parameter from the model.
            If `calctype` == "numerical", then try a numerical derivative
            If `calctype` == "taylor", evaluate the frequency with a Taylor series
        subtract_mean : bool or None, optional
            Subtract the mean of the residuals. This is ignored if the `PhaseOffset` component
            is present in the model. Default is to use the class attribute.
        use_weighted_mean : bool or None, optional
            Whether to use weighted mean for mean subtraction. Default is to use the class attribute.
        use_abs_phase : bool or None, optional
            Whether to use absolute phase (w.r.t. the TZR TOA). Default is to use the class attribute.

        Returns
        -------
        residuals : astropy.units.Quantity

        See Also
        --------
        :meth:`pint.residuals.Residuals.get_PSR_freq`
        :meth:`pint.residuals.Residuals.calc_phase_resids`
        :meth:`pint.residuals.Residuals.calc_whitened_resids`
        """
        assert calctype.lower() in ["modelf0", "taylor", "numerical"]
        if subtract_mean is None and use_weighted_mean is None:
            # if we are using the defaults, save the calculation
            if self.phase_resids is None:
                self.phase_resids = self.calc_phase_resids(
                    subtract_mean=subtract_mean,
                    use_weighted_mean=use_weighted_mean,
                    use_abs_phase=use_abs_phase,
                )
            phase_resids = self.phase_resids
        else:
            phase_resids = self.calc_phase_resids(
                subtract_mean=subtract_mean,
                use_weighted_mean=use_weighted_mean,
                use_abs_phase=use_abs_phase,
            )
        return (phase_resids / self.get_PSR_freq(calctype=calctype)).to(u.s)

    def calc_whitened_resids(self):
        """Compute whitened timing residuals (dimensionless).

        Whitened residuals are computed by subtracting the correlated
        noise realization from the time residuals and normalizes the
        result using scaled TOA uncertainties.

        This requires the `noise_resids` attribute to be set. This is
        usually available in a post-fit residuals object.

        Example usage::

            >>> ftr = Fitter.auto(toas, model)
            >>> ftr.fit_toas()
            >>> res = ftr.resids
            >>> white_res = res.calc_whitened_resids()

        See Also
        --------
        :meth:`pint.residuals.Residuals.calc_time_resids`
        :meth:`pint.residuals.Residuals.calc_phase_resids`
        """
        r = self.calc_time_resids()
        nr = sum(self.noise_resids.values())
        sigma = self.get_data_error()
        return ((r - nr) / sigma).to(u.dimensionless_unscaled)

    def _calc_gls_chi2(self, lognorm=False):
        """Compute the chi2 when correlated noise is present in the timing model.
        If the system is not singular, it uses Cholesky factorization to evaluate this.
        If the system is singular, it uses singular value decomposition instead.

        If `lognorm=True` is given, the log-normalization-factor of the likelihood
        function will also be returned.
        """

        assert self.model.has_correlated_errors

        s = self.time_resids.to_value(u.s)
        Ndiag = self.get_data_error().to_value(u.s) ** 2
        U = self.model.noise_model_designmatrix(self.toas)
        Phidiag = self.model.noise_model_basis_weight(self.toas)

        if "PHOFF" not in self.model.free_params:
            U = np.append(U, np.ones((len(self.toas), 1)), axis=1)
            Phidiag = np.append(Phidiag, [1e40])

        chi2, logdet_C = woodbury_dot(Ndiag, U, Phidiag, s, s)

        return (chi2, logdet_C / 2) if lognorm else chi2

    def _calc_ecorr_chi2(self, lognorm=False):
        """Compute the chi2 when ECORR is present in the timing
        model without any other correlated noise components."""

        assert (
            self.model.has_correlated_errors
            and not self.model.has_time_correlated_errors
            and "PHOFF" in self.model.free_params
        )

        m, t = self.model, self.toas

        ecorr_masks = m.components["EcorrNoise"].get_noise_basis(t).T.astype(bool)
        ecorr_weights = m.components["EcorrNoise"].get_noise_weights(t)

        r = self.time_resids
        sigma = self.get_data_error()

        # For TOAs which don't belong to any ECORR group.
        noecmask = np.logical_not(np.any(ecorr_masks, axis=0))
        Ndiag = sigma[noecmask] ** 2
        s = r[noecmask]
        chisq = (
            np.dot(s, s / Ndiag)
            if len(s) > 0
            else u.Quantity(0, u.dimensionless_unscaled)
        )
        logdet_C = np.sum(np.log(Ndiag.to_value(u.s**2)))

        # For TOAs which belong to an ECORR group.
        for ecmask, ecw in zip(ecorr_masks, ecorr_weights):
            Ndiag = sigma[ecmask] ** 2
            s = r[ecmask]

            v = np.ones(len(s)) << u.s

            chisq_sm, logdet_sm = sherman_morrison_dot(Ndiag, v, ecw, s, s)

            chisq += chisq_sm
            logdet_C += logdet_sm

        return (
            (chisq.to_value(u.dimensionless_unscaled), 0.5 * logdet_C)
            if lognorm
            else chisq.to_value(u.dimensionless_unscaled)
        )

    def _calc_wls_chi2(self, lognorm=False):
        """Compute the chi2 when no correlated noise components are present."""

        assert not self.model.has_correlated_errors

        # Residual units are in seconds. Error units are in microseconds.
        toa_errors = self.get_data_error()
        if (toa_errors == 0.0).any():
            return np.inf

        # The self.time_resids is in the unit of "s", the error "us".
        # This is more correct way, but it is the slowest.
        # return (((self.time_resids / self.toas.get_errors()).decompose()**2.0).sum()).value

        # This method is faster then the method above but not the most correct way
        # return ((self.time_resids.to(u.s) / self.toas.get_errors().to(u.s)).value**2.0).sum()

        # This the fastest way, but highly depend on the assumption of time_resids and
        # error units. Ensure only a pure number is returned.

        r = self.time_resids
        err = toa_errors.to(u.s)

        chi2 = ((r / err) ** 2.0).sum().value

        if not lognorm:
            return chi2

        log_norm = np.sum(np.log(err.value))
        return chi2, log_norm

    def calc_chi2(self, lognorm=False):
        """Return the weighted chi-squared for the model and toas.

        If the errors on the TOAs are independent this is a straightforward
        calculation, but if the noise model introduces correlated errors then
        obtaining a meaningful chi-squared value requires a Cholesky
        decomposition. This is carried out using the :method:`~pint.residuals.Residuals._calc_gls_chi2`
        helper function.

        The return value here is available as self.chi2, which will not
        redo the computation unless necessary.

        The chi-squared value calculated here is suitable for use in downhill
        minimization algorithms and Bayesian approaches.

        Handling of problematic results - degenerate conditions explored by
        a minimizer for example - may need to be checked to confirm that they
        correctly return infinity.

        If `lognorm=True` is given, the log-normalization-factor of the likelihood
        function will also be returned.

        Parameters
        ----------
        lognorm: bool
            If True, return the the log-normalization-factor of the likelihood
            function along with the chi2 value.

        Returns
        -------
        chi2                   if lognorm is False
        (chi2, log_norm)       if lognorm is True
        """

        if not self.model.has_correlated_errors:
            return self._calc_wls_chi2(lognorm=lognorm)
        elif (
            not self.model.has_time_correlated_errors
            and "PhaseOffset" in self.model.components
        ):
            return self._calc_ecorr_chi2(lognorm=lognorm)
        else:
            return self._calc_gls_chi2(lognorm=lognorm)

    def lnlikelihood(self):
        """Compute the log-likelihood for the model and TOAs."""
        chi2, log_norm = self.calc_chi2(lognorm=True)
        return -(chi2 / 2 + log_norm)

    def d_lnlikelihood_d_Ndiag(self):
        r = self.time_resids
        sigma = self.get_data_error()

        if not self.model.has_correlated_errors:
            Ndiag = sigma**2

            term1 = -(r**2) / Ndiag**2
            term2 = 1 / Ndiag

            return -0.5 * (term1 + term2)
        else:
            if (
                self.model.has_time_correlated_errors
                or "PHOFF" not in self.model.free_params
            ):
                raise NotImplementedError

            ecorr_masks = (
                self.model.components["EcorrNoise"]
                .get_noise_basis(self.toas)
                .T.astype(bool)
            )
            ecorr_weights = self.model.components["EcorrNoise"].get_noise_weights(
                self.toas
            )

            # For TOAs which don't belong to any ECORR group.
            noecmask = np.logical_not(np.any(ecorr_masks, axis=0))
            Ndiag = sigma[noecmask] ** 2
            s = r[noecmask]
            term1 = -(s**2) / Ndiag**2
            term2 = 1 / Ndiag
            term3 = term4 = 0

            # For TOAs which belong to an ECORR group.
            for ecmask, c in zip(ecorr_masks, ecorr_weights):
                Ndiag = sigma[ecmask] ** 2
                s = r[ecmask]
                v = np.ones(len(s)) << u.s

                s_Ninv_v = np.sum(s * v / Ndiag)
                v_Ninv_v = np.sum(v**2 / Ndiag)
                denom = 1 + c**2 * v_Ninv_v

                term1 += -(s**2) / Ndiag**2
                term2 += 1 / Ndiag
                term3 += 2 * c**2 * (
                    s_Ninv_v / denom * (s * v / Ndiag**2)
                ) - c**4 * (s_Ninv_v**2 / denom**2 * (v**2 / Ndiag**2))
                term4 += -(c**2) * (v**2 / Ndiag**2) / denom

            return -0.5 * (term1 + term2 + term3 + term4)

    def d_Ndiag_d_param(self, param):
        """Derivative of the white noise covariance matrix diagonal elements
        w.r.t. a white noise parameter (EFAC or EQUAD)."""
        sigma = self.get_data_error()
        d_sigma_d_param = self.model.d_toasigma_d_param(self.toas, param)
        return 2 * sigma * d_sigma_d_param

    def d_lnlikelihood_d_ECORR(self, param):
        par = self.model[param]

        fullmask = par.select_toa_mask(self.toas)
        t = self.toas[fullmask]

        sigma = self.get_data_error()
        r = self.time_resids
        c = par.quantity

        ecorr_masks = (
            self.model.components["EcorrNoise"].get_noise_basis(t).T.astype(bool)
        )

        result = 0

        # For TOAs which belong to an ECORR group.
        for ecmask in ecorr_masks:
            Ndiag = sigma[ecmask] ** 2
            s = r[ecmask]
            v = np.ones_like(s)

            s_Ninv_v = np.sum(s * v / Ndiag)
            v_Ninv_v = np.sum(v**2 / Ndiag)
            denom = 1 + c**2 * v_Ninv_v

            result += (
                c * (s_Ninv_v**2 - v_Ninv_v - c**2 * v_Ninv_v**2) / denom**2
            )

        return result

    def d_lnlikelihood_d_param(self, param):
        par = self.model[param]

        if self.model.has_correlated_errors and (
            self.model.has_time_correlated_errors
            or "PHOFF" not in self.model.free_params
        ):
            raise NotImplementedError

        if isinstance(par, maskParameter):
            if par.prefix in ["EFAC", "EQUAD"]:
                return np.sum(
                    self.d_lnlikelihood_d_Ndiag() * self.d_Ndiag_d_param(param)
                ).to(1 / par.units)
            elif par.prefix == "ECORR":
                return self.d_lnlikelihood_d_ECORR(param).to(1 / par.units)

        raise NotImplementedError(
            f"d_lnlikelihood_d_param is not defined for parameter {param}."
        )

    # def d_lnlikelihood_d_whitenoise_param(self, param):
    #     if self.model.has_correlated_errors:
    #         raise NotImplementedError(
    #             "d_lnlikelihood_d_whitenoise_param is not implemented "
    #             "for the case when correlated noise is present."
    #         )

    #     r = self.time_resids
    #     sigma = self.get_data_error()
    #     d_sigma_d_param = self.model.d_toasigma_d_param(self.toas, param)
    #     return np.sum(((r / sigma) ** 2 - 1) / sigma * d_sigma_d_param).to(1/self.model[param].units)

    def ecorr_average(self, use_noise_model=True):
        """Uses the ECORR noise model time-binning to compute "epoch-averaged" residuals.

        Requires ECORR be used in the timing model.  If
        ``use_noise_model`` is true, the noise model terms (EFAC, EQUAD, ECORR) will
        be applied to the TOA uncertainties, otherwise only the raw
        uncertainties will be used.

        Returns a dictionary with the following entries:

          mjds           Average MJD for each segment

          freqs          Average topocentric frequency for each segment

          time_resids    Average residual for each segment, time units

          noise_resids   Dictionary of per-noise-component average residual

          errors         Uncertainty on averaged residuals

          indices        List of lists giving the indices of TOAs in the original TOA table for each segment
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
            err = self.model.scaled_toa_uncertainty(self.toas)
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


class WidebandDMResiduals(Residuals):
    """Residuals for independent DM measurement (i.e. Wideband TOAs).

    This class manages the DM residuals from data that includes direct DM
    measurements associated with the TOAs.
    :class:`pint.residuals.WidebandTOAResiduals` combines one of these objects
    with a :class:`pint.residuals.Residuals` object.

    The values of interest are probably best accessed using the ``.resids``
    property, and the uncertainty using the ``.get_data_error()``.

    Attributes
    ----------
    dm_data : :class:`astropy.units.Quantity`
        The DM data extracted from the TOAs.
    dm_error : :class:`astropy.units.Quantity`
        The DM uncertainties extracted from the TOAs.

    Parameters
    ----------
    toas : :class:`pint.toa.TOAs`
        TOAs. They should include DM measurement data.
    model : :class:`pint.models.timing_model.TimingModel`
        The timing model.
    """

    def __init__(
        self,
        toas=None,
        model=None,
        residual_type="dm",
        unit=u.pc / u.cm**3,
        subtract_mean=False,
        use_weighted_mean=True,
    ):
        self.toas = toas
        self.model = model
        self.residual_type = residual_type
        self.unit = unit
        self.subtract_mean = subtract_mean
        self.use_weighted_mean = use_weighted_mean
        self.base_unit = u.pc / u.cm**3
        self.get_model_value = self.model.total_dm
        self.dm_data, self.dm_error, self.relevant_toas = self.get_dm_data()
        self._chi2 = None
        self._is_combined = False
        # For residual debugging
        self.debug_info = {}

    @property
    def resids(self):
        return self.calc_resids()

    @property
    def resids_value(self):
        """Get pure value of the residuals use the given base unit."""
        return self.resids.to_value(self.unit)

    @property
    def dof(self):
        """Return number of degrees of freedom for the DM model."""
        if self._is_combined:
            raise AttributeError(
                "Please use the `.dof` in the CombinedResidual"
                " class. The individual residual's dof is not "
                "calculated correctly in the combined residuals."
            )

        # only get dm type of model component
        # TODO provide a function in the timing model to get one type of component
        dof = len(self.dm_data) - sum(
            len(cp.free_params_component)
            for cp in self.model.components.values()
            if Dispersion in cp.__class__.__bases__
        )
        dof -= 1
        return dof

    def get_data_error(self, scaled=True):
        """Get data errors.

        Parameters
        ----------
        scaled: bool, optional
            If errors get scaled by the noise model.
        """
        return self.model.scaled_dm_uncertainty(self.toas) if scaled else self.dm_error

    def calc_resids(self):
        model_value = self.get_model_value(self.toas)[self.relevant_toas]
        resids = self.dm_data - model_value
        if self.subtract_mean:
            if self.use_weighted_mean:
                # Errs for weighted sum.  Units don't matter since they will
                # cancel out in the weighted sum.
                if self.dm_error is None or np.any(self.dm_error == 0):
                    raise ValueError(
                        "Some DM errors are zero - cannot calculate the "
                        "weighted residuals."
                    )
                wm = np.average(resids, weights=1.0 / (self.dm_error**2))
                resids -= wm
            else:
                resids -= resids.mean()
        return resids

    def calc_chi2(self):
        data_errors = self.get_data_error()
        if (data_errors == 0.0).any():
            return np.inf
        try:
            return ((self.resids / data_errors) ** 2.0).sum().decompose().value
        except ValueError:
            return ((self.resids / data_errors) ** 2.0).sum().decompose()

    def rms_weighted(self):
        """Compute weighted RMS of the residuals in time."""
        scaled_errors = self.get_data_error()
        if np.any(scaled_errors.value == 0):
            raise ValueError(
                "Some DM errors are zero - cannot calculate weighted RMS of residuals"
            )
        w = 1.0 / (scaled_errors**2)

        wmean, werr, wsdev = weighted_mean(self.resids, w, sdev=True)
        return wsdev

    def get_dm_data(self):
        """Get the independent measured DM data from TOA flags.

        This is to extract DM and uncertainty data from its representation in
        the flags on TOAs.

        FIXME: there should be a ``set_dm_data``  function.

        Returns
        -------
        valid_dm: `numpy.ndarray`
            Independent measured DM data from TOA line. It only returns the DM
            values that is present in the TOA flags.
        valid_error: `numpy.ndarray`
            The error associated with DM values in the TOAs.
        valid_index: list
            The TOA with DM data index.
        """
        dm_data, valid_data = self.toas.get_flag_value("pp_dm", as_type=float)
        dm_error, valid_error = self.toas.get_flag_value("pp_dme", as_type=float)
        if valid_data == []:
            raise ValueError("Input TOA object does not have wideband DM values")
        # Check valid error, if an error is none, change it to zero
        if valid_data != valid_error:
            raise ValueError("Input TOA object' DM data and DM errors do not match.")
        valid_dm = np.array(dm_data)[valid_data]
        valid_error = np.array(dm_error)[valid_error]
        return valid_dm * self.unit, valid_error * self.unit, valid_data

    def update_model(self, new_model, **kwargs):
        """Up date DM models from a new PINT timing model

        Parameters
        ----------
        new_model : `pint.timing_model.TimingModel`
        """

        self.model = new_model
        self.model_func = self.model.dm_value


residual_map = {"toa": Residuals, "dm": WidebandDMResiduals}


class CombinedResiduals:
    """Collect results from different type of residuals.

    Parameters
    ----------
    residuals: List of residual objects
        A list of different types of residual objects

    Note
    ----
    Since different type of residuals has different units, the overall
    residuals will have no units.
    """

    def __init__(self, residuals):
        self.residual_objs = collections.OrderedDict()
        for res in residuals:
            res._is_combined = True
            self.residual_objs[res.residual_type] = res
        # For residual debugging
        self.debug_info = {}

    @property
    def model(self):
        """Return the single timing model object."""
        raise AttributeError(
            "Combined residuals object does not provide a "
            "single timing model object. Please use the "
            "dedicated subclass."
        )

    @property
    def _combined_resids(self):
        """Residuals from all of the residual types."""
        all_resids = [res.resids_value for res in self.residual_objs.values()]
        return np.hstack(all_resids)

    @property
    def _combined_data_error(self):
        # Since it is the combined residual, the units are removed.
        dr = self.data_error
        return np.hstack([rv.value for rv in dr.values()])

    @property
    def unit(self):
        return {k: v.unit for k, v in self.residual_objs.items()}

    @property
    def chi2(self):
        return sum(res.chi2 for res in self.residual_objs.values())

    @property
    def data_error(self):
        errors = [
            (rs.residual_type, rs.get_data_error())
            for rs in self.residual_objs.values()
        ]
        return collections.OrderedDict(errors)

    def rms_weighted(self):
        """Compute weighted RMS of the residuals in time."""

        if np.any(self._combined_data_error == 0):
            raise ValueError(
                "Some data errors are zero - cannot calculate weighted RMS of residuals"
            )
        wrms = {}
        for rs in self.residual_objs.values():
            w = 1.0 / (rs.get_data_error() ** 2)
            wmean, werr, wsdev = weighted_mean(rs.resids, w, sdev=True)
            wrms[rs.residual_type] = wsdev
        return wrms


class WidebandTOAResiduals(CombinedResiduals):
    """A class for handling the wideband toa residuals.

    Wideband TOAs have independent measurement of DM values. The residuals for
    wideband TOAs have two parts, the TOA residuals and DM residuals. Both
    residuals will be used for fitting one timing model. Currently, the DM
    values are stored at the TOA object.

    The TOA and DM residuals are probably best accessed using the ``.toa`` and
    ``.dm`` properties.

    This class inherits the ``.chi2`` property from :class:`pint.residuals.CombinedResiduals`.

    Parameters
    ----------
    toas: :class:`pint.toa.TOAs`, optional
        The input TOAs object. Default: None
    model: :class:`pint.models.timing_model.TimingModel`, optional
        The input timing model. Default: None
    toa_resid_args: dict, optional
        The additional arguments(not including toas and model) for TOA residuals.
        Default: {}
    dm_resid_args: dict, optional
        The additional arguments(not including toas and model) for DM residuals.
        Default: {}
    """

    def __init__(self, toas, model, toa_resid_args={}, dm_resid_args={}):
        self.toas = toas
        self._model = model
        toa_resid = Residuals(
            self.toas, self.model, residual_type="toa", **toa_resid_args
        )
        dm_resid = Residuals(self.toas, self.model, residual_type="dm", **dm_resid_args)
        self._chi2 = None

        super().__init__([toa_resid, dm_resid])

    @property
    def toa(self):
        """Residuals object containing the TOA residuals."""
        return self.residual_objs["toa"]

    @property
    def dm(self):
        """WidebandDMResiduals object containing the DM residuals."""
        return self.residual_objs["dm"]

    @property
    def chi2(self):
        """Compute chi-squared as needed and cache the result."""
        if self._chi2 is None:
            self._chi2 = self.calc_chi2()
        assert self._chi2 is not None
        return self._chi2

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
        log.debug("Using wideband GLS fitter to compute residual chi2")
        # Use GLS but don't actually fit
        from pint.fitter import WidebandTOAFitter

        m = copy.deepcopy(self.model)
        m.free_params = []
        f = WidebandTOAFitter(
            self.toas, m, additional_args=dict(toa=dict(track_mode=self.toa.track_mode))
        )
        try:
            return f.fit_toas(maxiter=1, full_cov=full_cov)
        except LinAlgError as e:
            log.warning(
                "Degenerate conditions encountered when "
                "computing chi-squared: %s" % (e,)
            )
            return np.inf

    @property
    def model(self):
        """The model used to construct the residuals.

        Modifying this model, even changing its parameters, may have confusing
        effects. It is probably best to use :func:`copy.deepcopy` to duplicate
        it before making any changes.
        """
        return self._model

    @property
    def dof(self):
        """The number of degrees of freedom for the wideband residuals."""
        dof = len(self._combined_resids)
        dof -= len(self.model.free_params) + 1
        return dof

    @property
    def reduced_chi2(self):
        """Return the weighted reduced chi-squared."""
        return self.chi2 / self.dof
