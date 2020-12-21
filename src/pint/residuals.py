"""Objects for comparing models to data.

These objects can be constructed directly, as ``Residuals(toas, model)``, or
they are contructed during fitting operations with :class:`pint.fitter.Fitter`
objects, as ``fitter.residual``. Variants exist for arrival-time-only data
(:class:`pint.residuals.Residuals`) and for arrival times that come paired with
dispersion measures (:class:`pint.residuals.WidebandTOAResiduals`).
"""
from __future__ import absolute_import, division, print_function

import collections
import warnings

import astropy.units as u
import numpy as np
from astropy import log
from scipy.linalg import LinAlgError

from pint.models.dispersion_model import Dispersion
from pint.phase import Phase
from pint.utils import weighted_mean

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
    model: :class:`pint.models.timing_model.TimingModel`, optinonal
        Input model object. Default: None
    residual_type: str, optional
        The type of the resiudals. Default: 'toa'
    unit: :class:`astropy.units.Unit`, optional
        The defualt unit of the residuals. Default: u.s
    subtract_mean : bool
        Controls whether mean will be subtracted from the residuals
    use_weighted_mean : bool
        Controls whether mean compution is weighted (by errors) or not.
    track_mode : None, "nearest", "use_pulse_numbers"
        Controls how pulse numbers are assigned. ``"nearest"`` assigns
        each TOA to the nearest integer pulse. ``"use_pulse_numbers"`` uses the
        ``pulse_number`` column of the TOAs table to assign pulse numbers. If the
        default, None, is passed, use the pulse numbers if and only if the model has
        parameter TRACK == "-2".
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
    ):
        if cls is Residuals:
            try:
                cls = residual_map[residual_type.lower()]
            except KeyError:
                raise ValueError(
                    "'{}' is not a PINT supported residual. Currently "
                    "supported data types are {}".format(
                        residual_type, list(residual_map.keys())
                    )
                )

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
    ):
        self.toas = toas
        self.model = model
        self.residual_type = residual_type
        self.subtract_mean = subtract_mean
        self.use_weighted_mean = use_weighted_mean
        if track_mode is None:
            if getattr(self.model, "TRACK").value == "-2":
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
        # We should be carefully for the other type of residuals
        self.unit = unit
        # A flag to indentify if this residual object is combined with residual
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
        self.dof = self.dof

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
        warnings.warn("Please use reduced_chi2 instead.", DeprecationWarning)
        return self.chi2 / self.dof

    def get_data_error(self, scaled=True):
        """Get errors on time residuals.

        This returns the uncertainties on the time residuals, optionally scaled
        by the noise model.

        Parameter
        ---------
        scaled: bool, optional
            If errors get scaled by the noise model.
        """
        if not scaled:
            return self.toas.get_errors()
        else:
            return self.model.scaled_toa_uncertainty(self.toas)

    def rms_weighted(self):
        """Compute weighted RMS of the residals in time."""
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

    def get_PSR_freq(self, modelF0=True):
        """Return pulsar rotational frequency in Hz.

        Parameters
        ----------
        modelF0 : bool, optional
            If True, simply read the ``F0`` parameter from the model. If False,
            query the model for the spin period at the time of each TOA.

        Returns
        -------
        freq : Quantity
            Either the single ``F0`` in the model or the spin frequency at the moment of each TOA.
        """
        if modelF0:
            # TODO this function will be re-write and move to timing model soon.
            # The following is a temproary patch.
            if "Spindown" in self.model.components:
                F0 = self.model.F0.quantity
            elif "P0" in self.model.params:
                F0 = 1.0 / self.model.P0.quantity
            else:
                raise AttributeError(
                    "No pulsar spin parameter(e.g., 'F0'," " 'P0') found."
                )
            return F0.to(u.Hz)
        else:
            return self.model.d_phase_d_toa(self.toas)

    def calc_phase_resids(self):
        """Compute timing model residuals in pulse phase."""

        # Read any delta_pulse_numbers that are in the TOAs table.
        # These are for PHASE statements, -padd flags, as well as user-inserted phase jumps
        # Check for the column, and if not there then create it as zeros
        try:
            delta_pulse_numbers = Phase(self.toas.table["delta_pulse_number"])
        except IndexError:
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
        elif self.track_mode == "nearest":
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
        else:
            raise ValueError("Invalid track_mode '{}'".format(self.track_mode))
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
        """Compute timing model residuals in time (seconds)."""
        if self.phase_resids is None:
            self.phase_resids = self.calc_phase_resids()
        return (self.phase_resids / self.get_PSR_freq()).to(u.s)

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
            toa_errors = self.get_data_error()
            if (toa_errors == 0.0).any():
                return np.inf
            else:
                # The self.time_resids is in the unit of "s", the error "us".
                # This is more correct way, but it is the slowest.
                # return (((self.time_resids / self.toas.get_errors()).decompose()**2.0).sum()).value

                # This method is faster then the method above but not the most correct way
                # return ((self.time_resids.to(u.s) / self.toas.get_errors().to(u.s)).value**2.0).sum()

                # This the fastest way, but highly depend on the assumption of time_resids and
                # error units.
                # insure only a pure number returned
                try:
                    return ((self.time_resids / toa_errors.to(u.s)) ** 2.0).sum().value
                except ValueError:
                    return ((self.time_resids / toa_errors.to(u.s)) ** 2.0).sum()

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
        unit=u.pc / u.cm ** 3,
        subtract_mean=False,
        use_weighted_mean=True,
    ):

        self.toas = toas
        self.model = model
        self.residual_type = residual_type
        self.unit = unit
        self.subtract_mean = subtract_mean
        self.use_weighted_mean = use_weighted_mean
        self.base_unit = u.pc / u.cm ** 3
        self.get_model_value = self.model.total_dm
        self.dm_data, self.dm_error, self.relevant_toas = self.get_dm_data()
        self._chi2 = None
        self._is_combined = False

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
        dof = len(self.dm_data)
        # only get dm type of model component
        # TODO provide a function in the timing model to get one type of component
        for cp in self.model.components.values():
            if Dispersion in cp.__class__.__bases__:
                dof -= len(cp.free_params_component)
        dof -= 1
        return dof

    def get_data_error(self, scaled=True):
        """Get data errors.

        Parameter
        ---------
        scaled: bool, optional
            If errors get scaled by the noise model.
        """
        if not scaled:
            return self.dm_error
        else:
            return self.model.scaled_dm_uncertainty(self.toas)

    def calc_resids(self):
        model_value = self.get_model_value(self.toas)[self.relevant_toas]
        resids = self.dm_data - model_value
        if self.subtract_mean:
            if not self.use_weighted_mean:
                resids -= resids.mean()
            else:
                # Errs for weighted sum.  Units don't matter since they will
                # cancel out in the weighted sum.
                if self.dm_error is None or np.any(self.dm_error == 0):
                    raise ValueError(
                        "Some DM errors are zero - cannot calculate the "
                        "weighted residuals."
                    )
                wm = np.average(resids, weights=1.0 / (self.dm_error ** 2))
                resids -= wm
        return resids

    def calc_chi2(self):
        data_errors = self.get_data_error()
        if (data_errors == 0.0).any():
            return np.inf
        else:
            try:
                return ((self.resids / data_errors) ** 2.0).sum().decompose().value
            except ValueError:
                return ((self.resids / data_errors) ** 2.0).sum().decompose()

    def rms_weighted(self):
        """Compute weighted RMS of the residals in time."""
        scaled_errors = self.get_data_error()
        if np.any(scaled_errors.value == 0):
            raise ValueError(
                "Some DM errors are zero - cannot calculate weighted RMS of residuals"
            )
        w = 1.0 / (scaled_errors ** 2)

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
        dm_data, valid_data = self.toas.get_flag_value("pp_dm")
        dm_error, valid_error = self.toas.get_flag_value("pp_dme")
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
        A list of different typs of residual objects

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

    @property
    def model(self):
        """Return the single timing model object."""
        raise AttributeError(
            "Combined redisuals object does not provide a "
            "single timing model object. Pleaes use the "
            "dedicated subclass."
        )

    @property
    def _combined_resids(self):
        """Residuals from all of the residual types."""
        all_resids = []
        for res in self.residual_objs.values():
            all_resids.append(res.resids_value)
        return np.hstack(all_resids)

    @property
    def _combined_data_error(self):
        # Since it is the combinde residual, the units are removed.
        dr = self.data_error
        return np.hstack([rv.value for rv in dr.values()])

    @property
    def unit(self):
        units = {}
        for k, v in self.residual_objs.items():
            units[k] = v.unit
        return units

    @property
    def chi2(self):
        chi2 = 0
        for res in self.residual_objs.values():
            chi2 += res.chi2
        return chi2

    @property
    def data_error(self):
        errors = []
        for rs in self.residual_objs.values():
            errors.append((rs.residual_type, rs.get_data_error()))
        return collections.OrderedDict(errors)

    def rms_weighted(self):
        """Compute weighted RMS of the residals in time."""

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

    Parameter
    ---------
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
