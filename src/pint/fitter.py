"""Objects for managing the procedure of fitting models to TOAs.

The objects defined here can be used to set up a fitting problem and carry out
the fit, adjust the model, and repeat the fit as necessary.

The primary objects of interest will be :class:`pint.fitter.WLSFitter` for
basic fitting, :class:`pint.fitter.GLSFitter` for fitting with noise models
that imply correlated errors, and :class:`pint.fitter.WidebandTOAFitter` for
TOAs that contain DM information.  However, the Downhill fitter variants may offer better convergence.

Fitters in use::

    >>> fitter = WLSFitter(toas, model)
    >>> fitter.fit_toas()
    59.57431562376629118
    >>> fitter.print_summary()
    Fitted model using weighted_least_square method with 5 free parameters to 62 TOAs
    Prefit residuals Wrms = 1090.5802622239905 us, Postfit residuals Wrms = 21.182038012901092 us
    Chisq = 59.574 for 56 d.o.f. for reduced Chisq of 1.064

    PAR                        Prefit                  Postfit            Units
    =================== ==================== ============================ =====
    PSR                           1748-2021E                              None
    EPHEM                              DE421                              None
    UNITS                                TDB                              None
    START                                                         53478.3 d
    FINISH                                                        54187.6 d
    POSEPOCH                           53750                              d
    PX                                     0                              mas
    RAJ                         17h48m52.75s    17h48m52.8003s +/- 0.00014 hourangle_second
    DECJ                          -20d21m29s   -20d21m29.3833s +/- 0.033 arcsec
    PMRA                                   0                              mas / yr
    PMDEC                                  0                              mas / yr
    F0                               61.4855          61.485476554373(18) Hz
    F1                            -1.181e-15            -1.1813(14)x10-15 Hz / s
    PEPOCH                             53750                              d
    CORRECT_TROPOSPHERE                    N                              None
    PLANET_SHAPIRO                         N                              None
    NE_SW                                  0                              1 / cm3
    SWM                                    0
    DM                                 223.9                  224.114(35) pc / cm3
    DM1                                    0                              pc / (cm3 yr)
    TZRMJD                           53801.4                              d
    TZRSITE                                1                              None
    TZRFRQ                           1949.61                              MHz

    Derived Parameters:
    Period = 0.01626400340437608 s +/- 4.784091376048965e-15 s
    Pdot = 3.1248325489308735e-19 +/- 3.8139606793005067e-22
    Characteristic age = 8.246e+08 yr (braking index = 3)
    Surface magnetic field = 2.28e+09 G
    Magnetic field at light cylinder = 4806 G
    Spindown Edot = 2.868e+33 erg / s (I=1e+45 cm2 g)

To automatically select a fitter based on the properties of the data and model::

    >>> fitter = Fitter.auto(toas, model)

"""

import contextlib
import copy
from warnings import warn

import astropy.units as u
import numpy as np
import scipy.linalg
import scipy.optimize as opt
from loguru import logger as log
from numdifftools import Hessian

import pint
import pint.utils
import pint.derived_quantities
from pint.models.parameter import (
    AngleParameter,
    boolParameter,
    strParameter,
)
from pint.pint_matrix import (
    CorrelationMatrix,
    CovarianceMatrix,
    CovarianceMatrixMaker,
    DesignMatrixMaker,
    combine_covariance_matrix,
    combine_design_matrices_by_param,
    combine_design_matrices_by_quantity,
)
from pint.residuals import Residuals, WidebandTOAResiduals
from pint.toa import TOAs
from pint.utils import FTest, normalize_designmatrix


__all__ = [
    "Fitter",
    "WLSFitter",
    "GLSFitter",
    "WidebandTOAFitter",
    "PowellFitter",
    "DownhillFitter",
    "DownhillWLSFitter",
    "DownhillGLSFitter",
    "WidebandDownhillFitter",
    "WidebandLMFitter",
    "ConvergenceFailure",
    "StepProblem",
    "MaxiterReached",
]

try:
    from functools import cached_property
except ImportError:
    # not supported in python 3.7
    # This is just the code from python 3.8
    from _thread import RLock

    _NOT_FOUND = object()

    class cached_property:
        def __init__(self, func):
            self.func = func
            self.attrname = None
            self.__doc__ = func.__doc__
            self.lock = RLock()

        def __set_name__(self, owner, name):
            if self.attrname is None:
                self.attrname = name
            elif name != self.attrname:
                raise TypeError(
                    "Cannot assign the same cached_property to two different names "
                    f"({self.attrname!r} and {name!r})."
                )

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            if self.attrname is None:
                raise TypeError(
                    "Cannot use cached_property instance without calling __set_name__ on it."
                )
            try:
                cache = instance.__dict__
            except AttributeError:
                # not all objects have __dict__ (e.g. class defines slots)
                msg = (
                    f"No '__dict__' attribute on {type(instance).__name__!r} "
                    f"instance to cache {self.attrname!r} property."
                )
                raise TypeError(msg) from None
            val = cache.get(self.attrname, _NOT_FOUND)
            if val is _NOT_FOUND:
                with self.lock:
                    # check if another thread filled cache while we awaited lock
                    val = cache.get(self.attrname, _NOT_FOUND)
                    if val is _NOT_FOUND:
                        val = self.func(instance)
                        try:
                            cache[self.attrname] = val
                        except TypeError:
                            msg = (
                                f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                                f"does not support item assignment for caching {self.attrname!r} property."
                            )
                            raise TypeError(msg) from None
            return val


class DegeneracyWarning(UserWarning):
    pass


class ConvergenceFailure(ValueError):
    pass


class MaxiterReached(ConvergenceFailure):
    pass


class StepProblem(ConvergenceFailure):
    pass


class Fitter:
    """Base class for objects encapsulating fitting problems.

    The fitting function should be defined as the fit_toas() method.

    The Fitter object makes a :func:`copy.deepcopy` of the model and stores it
    in the `.model` attribute. This is the model used for fitting, and it can
    be modified, for example by freezing or thawing parameters
    (``fitter.model.F0.frozen = False``). When
    ``.fit_toas()`` is executed this model will be updated to reflect the
    results of the fitting process.

    The Fitter also caches a copy of the original model so it can be restored
    with ``reset_model()``.

    Try :func:`pint.fitter.Fitter.auto` to automatically get the appropriate fitter type

    Attributes
    ----------
    model : :class:`pint.models.timing_model.TimingModel`
        The model the fitter is working on. Once ``fit_toas()`` has been run,
        this model will be modified to reflect the results.
    model_init : :class:`pint.models.timing_model.TimingModel`
        The initial, prefit model.

    Parameters
    ----------
    toas : a pint TOAs instance
        The input toas.
    model : a pint timing model instance
        The initial timing model for fitting.
    track_mode : str, optional
        How to handle phase wrapping. This is used when creating
        :class:`pint.residuals.Residuals` objects, and its meaning is defined there.
    residuals : :class:`pint.residuals.Residuals`
        Initial residuals. This argument exists to support an optimization, where
        ``GLSFitter`` is used to compute ``chi2`` for appropriate Residuals objects.
    """

    def __init__(self, toas, model, track_mode=None, residuals=None):
        if not set(model.free_params).issubset(model.fittable_params):
            free_unfittable_params = set(model.free_params).difference(
                model.fittable_params
            )
            raise ValueError(
                f"Cannot create fitter because the following unfittable parameters "
                f"were found unfrozen in the model: {free_unfittable_params}. "
                f"Freeze these parameters before creating the fitter."
            )

        self.toas = toas
        self.model_init = model
        self.track_mode = track_mode
        if residuals is None:
            self.resids_init = self.make_resids(self.model_init)
        else:
            # residuals were provided, we're just going to use them
            self.resids_init = residuals
            # probably using GLSFitter to compute a chi-squared
        self.model = copy.deepcopy(self.model_init)
        self.resids = copy.deepcopy(self.resids_init)
        self.fitresult = []
        self.method = None
        self.is_wideband = False
        self.converged = False

    @classmethod
    def auto(
        cls, toas, model, downhill=True, track_mode=None, residuals=None, **kwargs
    ):
        """Automatically return the proper :class:`pint.fitter.Fitter` object depending on the TOAs and model.

        In general the `downhill` fitters are to be preferred.
        See https://github.com/nanograv/PINT/wiki/How-To#choose-a-fitter for the logic used.

        Parameters
        ----------
        toas : a pint TOAs instance
            The input toas.
        model : a pint timing model instance
            The initial timing model for fitting.
        downhill : bool, optional
            Whether or not to use the downhill fitter variant
        track_mode : str, optional
            How to handle phase wrapping. This is used when creating
            :class:`pint.residuals.Residuals` objects, and its meaning is defined there.
        residuals : :class:`pint.residuals.Residuals`
            Initial residuals. This argument exists to support an optimization, where
            ``GLSFitter`` is used to compute ``chi2`` for appropriate Residuals objects.

        Returns
        -------
        :class:`pint.fitter.Fitter`
            Returns appropriate subclass
        """
        if toas.wideband:
            if downhill:
                log.info(
                    "For wideband TOAs and downhill fitter, returning 'WidebandDownhillFitter'"
                )
                return WidebandDownhillFitter(
                    toas, model, track_mode=track_mode, residuals=residuals, **kwargs
                )
            else:
                log.info(
                    "For wideband TOAs and non-downhill fitter, returning 'WidebandTOAFitter'"
                )
                return WidebandTOAFitter(toas, model, track_mode=track_mode, **kwargs)
        elif model.has_correlated_errors:
            if downhill:
                log.info(
                    "For narrowband TOAs with correlated errors and downhill fitter, returning 'DownhillGLSFitter'"
                )
                return DownhillGLSFitter(
                    toas,
                    model,
                    track_mode=track_mode,
                    residuals=residuals,
                    **kwargs,
                )
            else:
                log.info(
                    "For narrowband TOAs with correlated errors and non-downhill fitter, returning 'GLSFitter'"
                )
                return GLSFitter(
                    toas,
                    model,
                    track_mode=track_mode,
                    residuals=residuals,
                    **kwargs,
                )
        elif downhill:
            log.info(
                "For narrowband TOAs without correlated errors and downhill fitter, returning 'DownhillWLSFitter'"
            )
            return DownhillWLSFitter(
                toas,
                model,
                track_mode=track_mode,
                residuals=residuals,
                **kwargs,
            )
        else:
            log.info(
                "For narrowband TOAs without correlated errors and non-downhill fitter, returning 'WLSFitter'"
            )
            return WLSFitter(
                toas,
                model,
                track_mode=track_mode,
                residuals=residuals,
                **kwargs,
            )

    def fit_toas(self, maxiter=None, debug=False):
        """Run fitting operation.

        This method needs to be implemented by subclasses. All implementations
        should call ``self.model.validate()`` and
        ``self.model.validate_toas()`` before doing the fitting.
        """
        raise NotImplementedError

    def get_summary(self, nodmx=False):
        """Return a human-readable summary of the Fitter results.

        Parameters
        ----------
        nodmx : bool
            Set to True to suppress printing DMX parameters in summary
        """

        # Need to check that fit has been done first!
        if not hasattr(self, "parameter_covariance_matrix"):
            log.warning(
                "fit_toas() has not been run, so pre-fit and post-fit will be the same!"
            )

        from uncertainties import ufloat

        # Check if Wideband or not
        is_wideband = self.is_wideband

        # First, print fit quality metrics
        s = f"Fitted model using {self.method} method with {len(self.model.free_params)} free parameters to {self.toas.ntoas} TOAs\n"
        if is_wideband:
            s += f"Prefit TOA residuals Wrms = {self.resids_init.toa.rms_weighted()}, Postfit TOA residuals Wrms = {self.resids.toa.rms_weighted()}\n"
            s += f"Prefit DM residuals Wrms = {self.resids_init.dm.rms_weighted()}, Postfit DM residuals Wrms = {self.resids.dm.rms_weighted()}\n"
        else:
            s += f"Prefit residuals Wrms = {self.resids_init.rms_weighted()}, Postfit residuals Wrms = {self.resids.rms_weighted()}\n"
        s += f"Chisq = {self.resids.chi2:.3f} for {self.resids.dof} d.o.f. for reduced Chisq of {self.resids.reduced_chi2:.3f}\n"
        s += "\n"

        # to handle all parameter names, determine the longest length for the first column
        longestName = 0  # optionally specify the minimum length here instead of 0
        for pn in self.model.params:
            if nodmx and pn.startswith("DMX"):
                continue
            if len(pn) > longestName:
                longestName = len(pn)
        # convert to a string to insert before the format call
        spacingName = str(longestName)

        # Next, print the model parameters
        s += ("{:<" + spacingName + "s} {:^20s} {:^28s} {}\n").format(
            "PAR", "Prefit", "Postfit", "Units"
        )
        s += ("{:<" + spacingName + "s} {:>20s} {:>28s} {}\n").format(
            "=" * longestName, "=" * 20, "=" * 28, "=" * 5
        )
        for pn in self.model.params:
            if nodmx and pn.startswith("DMX"):
                continue
            prefitpar = getattr(self.model_init, pn)
            par = getattr(self.model, pn)
            if par.value is not None:
                if isinstance(par, strParameter):
                    s += ("{:" + spacingName + "s} {:>20s} {:28s} {}\n").format(
                        pn,
                        prefitpar.value if prefitpar.value is not None else "",
                        par.value,
                        par.units,
                    )
                elif isinstance(par, AngleParameter):
                    # Add special handling here to put uncertainty into arcsec
                    if par.frozen:
                        s += ("{:" + spacingName + "s} {:>20s} {:>28s} {} \n").format(
                            pn, str(prefitpar.quantity), "", par.units
                        )
                    else:
                        uncertainty_unit = (
                            pint.hourangle_second
                            if par.units == u.hourangle
                            else u.arcsec
                        )
                        s += (
                            "{:" + spacingName + "s} {:>20s}  {:>16s} +/- {:.2g} \n"
                        ).format(
                            pn,
                            str(prefitpar.quantity),
                            str(par.quantity),
                            par.uncertainty.to(uncertainty_unit),
                        )
                elif isinstance(par, boolParameter):
                    s += ("{:" + spacingName + "s} {:>20s} {:28s} {}\n").format(
                        pn, prefitpar.str_quantity(prefitpar.value), "", par.units
                    )
                elif par.frozen:
                    if par.name in ["START", "FINISH"] and prefitpar.value is None:
                        s += ("{:" + spacingName + "s} {:20s} {:28g} {} \n").format(
                            pn, " ", par.value, par.units
                        )
                    elif par.name in ["START", "FINISH"]:
                        s += ("{:" + spacingName + "s} {:20g} {:28g} {} \n").format(
                            pn, prefitpar.value, par.value, par.units
                        )
                    elif (
                        par.name in ["CHI2", "CHI2R", "TRES", "DMRES"]
                        and prefitpar.value is None
                    ):
                        s += ("{:" + spacingName + "s} {:20s} {:28g} {} \n").format(
                            pn, " ", par.value, par.units
                        )
                    elif par.name in ["CHI2", "CHI2R", "TRES", "DMRES"]:
                        s += ("{:" + spacingName + "s} {:20g} {:28g} {} \n").format(
                            pn, prefitpar.value, par.value, par.units
                        )
                    else:
                        s += ("{:" + spacingName + "s} {:20g} {:28s} {} \n").format(
                            pn, prefitpar.value, "", par.units
                        )

                else:
                    # s += "{:14s} {:20g} {:20g} {:20.2g} {} \n".format(
                    #     pn,
                    #     prefitpar.value,
                    #     par.value,
                    #     par.uncertainty.value,
                    #     par.units,
                    # )
                    s += ("{:" + spacingName + "s} {:20g} {:28SP} {} \n").format(
                        pn,
                        prefitpar.value,
                        ufloat(par.value, par.uncertainty.value),
                        par.units,
                    )
        s += "\n" + self.model.get_derived_params()
        return s

    def get_derived_params(self, returndict=False):
        """Return a string with various derived parameters from the fitted model

        Parameters
        ----------
        returndict : bool, optional
            Whether to only return the string of results or also a dictionary

        Returns
        -------
        results : str
        parameters : dict, optional

        See Also
        --------
        :func:`pint.models.timing_model.TimingModel.get_derived_params`
        """

        return self.model.get_derived_params(
            rms=self.resids.toa.rms_weighted()
            if self.is_wideband
            else self.resids.rms_weighted(),
            ntoas=self.toas.ntoas,
            returndict=returndict,
        )

    def print_summary(self):
        """Write a summary of the TOAs to stdout."""
        print(self.get_summary())

    def plot(self):
        """Make residuals plot.

        This produces a time residual plot.
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support

        quantity_support()
        fig, ax = plt.subplots(figsize=(16, 9))
        mjds = self.toas.get_mjds()
        ax.errorbar(mjds, self.resids.time_resids, yerr=self.toas.get_errors(), fmt="+")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Residuals")
        try:
            psr = self.model.PSR
        except AttributeError:
            psr = self.model.PSRJ
        else:
            psr = "Residuals"
        ax.set_title(psr)
        ax.grid(True)
        plt.show()

    def update_model(self, chi2=None):
        """Update the model to reflect fit results and TOA properties.

        This is called by ``fit_toas`` to ensure that parameters like
        ``START``, ``FINISH``, ``EPHEM``, and ``DMDATA`` are set in the model
        to reflect the TOAs in actual use.
        """
        self.model.START.value = self.toas.first_MJD
        self.model.FINISH.value = self.toas.last_MJD
        self.model.NTOA.value = len(self.toas)
        self.model.EPHEM.value = self.toas.ephem
        self.model.DMDATA.value = hasattr(self.resids, "dm")
        self.model.CLOCK.value = (
            f"TT({self.toas.clock_corr_info['bipm_version']})"
            if self.toas.clock_corr_info["include_bipm"]
            else "TT(TAI)"
        )
        if chi2 is not None:
            # assume a fit has been done
            self.model.CHI2.value = chi2
            self.model.CHI2R.value = chi2 / self.resids.dof
            if not self.is_wideband:
                self.model.TRES.quantity = self.resids.rms_weighted()
            else:
                self.model.TRES.quantity = self.resids.rms_weighted()["toa"]
                self.model.DMRES.quantity = self.resids.rms_weighted()["dm"]

    def reset_model(self):
        """Reset the current model to the initial model."""
        self.model = copy.deepcopy(self.model_init)
        self.update_resids()
        self.fitresult = []

    def update_resids(self):
        """Update the residuals.

        Run after updating a model parameter.
        """
        self.resids = self.make_resids(self.model)

    def make_resids(self, model):
        return Residuals(toas=self.toas, model=model, track_mode=self.track_mode)

    def get_designmatrix(self):
        """Return the model's design matrix for these TOAs."""
        return self.model.designmatrix(toas=self.toas, incfrozen=False, incoffset=True)

    def _get_corr_cov_matrix(
        self, matrix_type, with_phase, pretty_print, prec, usecolor
    ):
        if hasattr(self, f"parameter_{matrix_type}_matrix"):
            cm = getattr(self, f"parameter_{matrix_type}_matrix")
            if not pretty_print:
                return cm.prettyprint(prec=prec, offset=with_phase)
            else:
                print(cm.prettyprint(prec=prec, offset=with_phase, usecolor=usecolor))
        else:
            log.error(
                f"You must run .fit_toas() before accessing the {matrix_type} matrix"
            )
            raise AttributeError

    def get_parameter_covariance_matrix(
        self, with_phase=False, pretty_print=False, prec=3
    ):
        """Show the parameter covariance matrix post-fit.

        If with_phase, then show and return the phase column as well.
        If pretty_print, then also pretty-print on stdout the matrix.
        prec is the precision of the floating point results.
        """
        return self._get_corr_cov_matrix(
            "covariance", with_phase, pretty_print, prec, "False"
        )

    def get_parameter_correlation_matrix(
        self, with_phase=False, pretty_print=False, prec=3, usecolor=True
    ):
        """Show the parameter correlation matrix post-fit.

        If with_phase, then show and return the phase column as well.
        If pretty_print, then also pretty-print on stdout the matrix.
        prec is the precision of the floating point results. If
        usecolor is True, then pretty printing will have color.
        """
        return self._get_corr_cov_matrix(
            "correlation", with_phase, pretty_print, prec, usecolor
        )

    def ftest(self, parameter, component, remove=False, full_output=False, maxiter=1):
        """Compare the significance of adding/removing parameters to a timing model.

        Parameters
        -----------
        parameter : PINT parameter object
            (may be a list of parameter objects)
        component : String
            Name of component of timing model that the parameter should be added to (may be a list)
            The number of components must equal number of parameters.
        remove : Bool
            If False, will add the listed parameters to the model. If True will remove the input
            parameters from the timing model.
        full_output : Bool
            If False, just returns the result of the F-Test. If True, will also return the new
            model's residual RMS (us), chi-squared, and number of degrees of freedom of
            new model.
        maxiter : int
            How many times to run the linear least-squares fit, re-evaluating
            the derivatives at each step for the F-tested model. Default is one.

        Returns
        --------
        dictionary

            ft : Float
                F-test significance value for the model with the larger number of
                components over the other. Computed with pint.utils.FTest().

            resid_rms_test : Float (Quantity)
                If full_output is True, returns the RMS of the residuals of the tested model
                fit. Will be in units of microseconds as an astropy quantity. If wideband fitter
                this will be the time residuals.

            resid_wrms_test : Float (Quantity)
                If full_output is True, returns the Weighted RMS of the residuals of the tested model
                fit. Will be in units of microseconds as an astropy quantity. If wideband fitter
                this will be the time residuals.

            chi2_test : Float
                If full_output is True, returns the chi-squared of the tested model. If wideband
                fitter this will be the total chi-squared of the combined residual.

            dof_test : Int
                If full_output is True, returns the degrees of freedom of the tested model.
                If wideband fitter this will be the total chi-squared of the combined residual.

            dm_resid_rms_test : Float (Quantity)
                If full_output is True and a wideband timing fitter is used, returns the
                RMS of the DM residuals of the tested model fit. Will be in units of
                pc/cm^3 as an astropy quantity.

            dm_resid_wrms_test : Float (Quantity)
                If full_output is True and a wideband timing fitter is used, returns the
                Weighted RMS of the DM residuals of the tested model fit. Will be in units of
                pc/cm^3 as an astropy quantity.
        """
        # Check if Wideband or not
        NB = not self.is_wideband
        # Copy the fitter that we do not change the initial model and fitter
        fitter_copy = copy.deepcopy(self)
        # We need the original degrees of freedom and chi-squared value
        # Because this applies to nested models, model 1 must always have fewer parameters
        if remove:
            dof_2 = self.resids.dof
            chi2_2 = self.resids.chi2
        else:
            dof_1 = self.resids.dof
            chi2_1 = self.resids.chi2
        # Single inputs are converted to lists to handle arb. number of parameters
        if type(parameter) is not list:
            parameter = [parameter]
        # also do the components
        if type(component) is not list:
            component = [component]
        # if not the same length, exit with error
        if len(parameter) != len(component):
            raise RuntimeError(
                "Number of input parameters must match number of input components."
            )
        # Now check if we want to remove or add components; start with removing
        if remove:
            # Set values to zero and freeze them
            for p in parameter:
                getattr(fitter_copy.model, "{:}".format(p.name)).value = 0.0
                getattr(fitter_copy.model, "{:}".format(p.name)).uncertainty_value = 0.0
                getattr(fitter_copy.model, "{:}".format(p.name)).frozen = True
            # validate and setup model
            fitter_copy.model.validate()
            fitter_copy.model.setup()
            # Now refit
            fitter_copy.fit_toas(maxiter=maxiter)
            # FIXME: check convergence
            # Now get the new values
            dof_1 = fitter_copy.resids.dof
            chi2_1 = fitter_copy.resids.chi2
        else:
            # Dictionary of parameters to check to makes sure input value isn't zero
            check_params = {
                "M2": 0.25,
                "SINI": 0.8,
                "PB": 10.0,
                "T0": 54000.0,
                "FB0": 1.1574e-6,
            }
            # Add the parameters
            for ii in range(len(parameter)):
                # Check if parameter already exists in model
                if hasattr(fitter_copy.model, "{:}".format(parameter[ii].name)):
                    # Set frozen to False
                    getattr(
                        fitter_copy.model, "{:}".format(parameter[ii].name)
                    ).frozen = False
                    # Check if parameter is one that needs to be checked
                    if (
                        parameter[ii].name in check_params
                        and parameter[ii].value == 0.0
                    ):
                        log.warning(
                            f"Default value for {parameter[ii].name} cannot be 0, resetting to {check_params[parameter[ii].name]}"
                        )
                        parameter[ii].value = check_params[parameter[ii].name]
                    getattr(
                        fitter_copy.model, "{:}".format(parameter[ii].name)
                    ).value = parameter[ii].value
                else:
                    fitter_copy.model.components[component[ii]].add_param(
                        parameter[ii], setup=True
                    )
            # validate and setup model
            fitter_copy.model.validate()
            fitter_copy.model.setup()
            # Now refit
            fitter_copy.fit_toas(maxiter=maxiter)
            # FIXME: check convergence
            # Now get the new values
            dof_2 = fitter_copy.resids.dof
            chi2_2 = fitter_copy.resids.chi2
        # Now run the actual F-test
        ft = FTest(chi2_1, dof_1, chi2_2, dof_2)

        if not full_output:
            return {"ft": ft}
        if remove:
            dof_test = dof_1
            chi2_test = chi2_1
        else:
            dof_test = dof_2
            chi2_test = chi2_2
        if NB:
            resid_rms_test = fitter_copy.resids.time_resids.std().to(u.us)
            resid_wrms_test = fitter_copy.resids.rms_weighted()  # units: us
            return {
                "ft": ft,
                "resid_rms_test": resid_rms_test,
                "resid_wrms_test": resid_wrms_test,
                "chi2_test": chi2_test,
                "dof_test": dof_test,
            }
        else:
            # Return the dm and time resid values separately
            resid_rms_test = fitter_copy.resids.toa.time_resids.std().to(u.us)
            resid_wrms_test = fitter_copy.resids.toa.rms_weighted()  # units: us
            dm_resid_rms_test = fitter_copy.resids.dm.resids.std()
            dm_resid_wrms_test = fitter_copy.resids.dm.rms_weighted()
            return {
                "ft": ft,
                "resid_rms_test": resid_rms_test,
                "resid_wrms_test": resid_wrms_test,
                "chi2_test": chi2_test,
                "dof_test": dof_test,
                "dm_resid_rms_test": dm_resid_rms_test,
                "dm_resid_wrms_test": dm_resid_wrms_test,
            }

    def minimize_func(self, x, *args):
        """Wrapper function for the residual class.

        This is meant to be passed to
        ``scipy.optimize.minimize``. The function must take a single list of input
        values, x, and a second optional tuple of input arguments.  It returns
        a quantity to be minimized (in this case chi^2).
        """
        self.set_params(dict(zip(args, x)))
        self.update_resids()
        # Return chi^2
        return self.resids.chi2

    def get_params_dict(self, which="free", kind="quantity"):
        """Return a dict mapping parameter names to values.

        See :func:`pint.models.timing_model.TimingModel.get_params_dict`.
        """
        return self.model.get_params_dict(which=which, kind=kind)

    def set_fitparams(self, *params):
        """Update the "frozen" attribute of model parameters. Deprecated."""
        warn(
            "This function is confusing and deprecated. Set self.model.free_params instead.",
            category=DeprecationWarning,
        )
        # TODO, maybe reconsider for the input?
        fit_params_name = []
        if isinstance(params[0], (list, tuple)):
            params = params[0]
        for pn in params:
            if pn in self.model.params:
                fit_params_name.append(pn)
            else:
                rn = self.model.match_param_aliases(pn)
                if rn != "":
                    fit_params_name.append(rn)
                else:
                    raise ValueError(f"Unrecognized parameter {pn}")
        self.model.fit_params = fit_params_name

    def get_allparams(self):
        """Return a dict of all param names and values. Deprecated."""
        warn(
            "This function is confusing and deprecated. Use self.model.get_params_dict.",
            category=DeprecationWarning,
        )
        return self.model.get_params_dict("all", "quantity")

    def get_fitparams(self):
        """Return a dict of fittable param names and quantity. Deprecated."""
        warn(
            "This function is confusing and deprecated. Use self.model.get_params_dict.",
            category=DeprecationWarning,
        )
        return self.model.get_params_dict("free", "quantity")

    def get_fitparams_num(self):
        """Return a dict of fittable param names and numeric values. Deprecated."""
        warn(
            "This function is confusing and deprecated. Use self.model.get_params_dict.",
            category=DeprecationWarning,
        )
        return self.model.get_params_dict("free", "num")

    def get_fitparams_uncertainty(self):
        """Return a dict of fittable param names and numeric values. Deprecated."""
        warn(
            "This function is confusing and deprecated. Use self.model.get_params_dict.",
            category=DeprecationWarning,
        )
        return self.model.get_params_dict("free", "uncertainty")

    def set_params(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        See :func:`pint.models.timing_model.TimingModel.set_param_values`.
        """
        self.model.set_param_values(fitp)

    def set_param_uncertainties(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        See :func:`pint.models.timing_model.TimingModel.set_param_uncertainties`.
        """
        self.model.set_param_uncertainties(fitp)

    @property
    def covariance_matrix(self):
        warn(
            "This parameter is deprecated. Use `parameter_covariance_matrix` instead of `covariance_matrix`",
            category=DeprecationWarning,
        )
        return self.parameter_covariance_matrix


class InvalidModelParameters(ValueError):
    pass


class CorrelatedErrors(ValueError):
    def __init__(self, model):
        trouble_components = [
            c.__class__.__name__
            for c in model.NoiseComponent_list
            if c.introduces_correlated_errors
        ]
        super().__init__(
            f"Model has correlated errors and requires a GLS-based fitter; "
            f"remove {trouble_components} if you want to use WLS"
        )
        self.trouble_components = trouble_components


class ModelState:
    """Record a model state and cache calculations

    This class keeps track of a particular model state and all the associated
    matrices - design matrices, singular value decompositions, what have you -
    that are needed to compute a step and evaluate the quality of the fit.

    These objects should be regarded as immutable but lazily evaluated.
    """

    def __init__(self, fitter, model):
        self.fitter = fitter
        self.model = model

    @cached_property
    def resids(self):
        try:
            return self.fitter.make_resids(self.model)
        except ValueError as e:
            raise InvalidModelParameters("Step landed at invalid point") from e

    @cached_property
    def chi2(self):
        # there may be some shareable computation here
        try:
            return self.resids.chi2
        except ValueError as e:
            raise InvalidModelParameters("Cannot compute chi2") from e

    @cached_property
    def step(self):
        raise NotImplementedError

    @cached_property
    def parameter_covariance_matrix(self):
        raise NotImplementedError

    @property
    def covariance_matrix(self):
        warn(
            "This parameter is deprecated.  Use `parameter_covariance_matrix` instead of `covariance_matrix`",
            category=DeprecationWarning,
        )
        return self.parameter_covariance_matrix

    def predicted_chi2(self, step, lambda_):
        """Predict the chi2 after taking a step based on the linear approximation"""
        raise NotImplementedError

    def take_step_model(self, step, lambda_=1):
        """Make a new model reflecting the new parameters."""
        # log.debug(f"Taking step {lambda_} * {list(zip(self.params, step))}")
        new_model = copy.deepcopy(self.model)
        for p, s in zip(self.params, step * lambda_):
            try:
                with contextlib.suppress(ValueError):
                    log.trace(f"Adjusting {getattr(self.model, p)} by {s}")
                pm = getattr(new_model, p)
                if pm.value is None:
                    pm.value = 0
                pm.value += s
                # getattr(new_model, p).value = getattr(self.model, p).value + s
                # getattr(self.model, p) + s
                # getattr(new_model, p).value = s
            except AttributeError:
                if p != "Offset":
                    log.warning(f"Unexpected parameter {p}")
        return new_model

    def take_step(self, step, lambda_):
        """Return a new state moved by lambda_*step."""
        raise NotImplementedError


class DownhillFitter(Fitter):
    """Abstract base class for downhill fitters.

    These fitters use the algorithm implemented here, in
    :func:`pint.fitter.DownhillFitter.fit_toas` to work their way towards a
    solution, keeping track of convergence. The linear algebra required by
    various kinds of fitting is abstracted away into
    :class:`pint.fitter.ModelState` objects so that this same code can be used
    for correlated or uncorrelated TOA errors and narrowband or wideband TOAs.
    """

    def __init__(self, toas, model, track_mode=None, residuals=None):
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.method = "downhill_checked"

    def _fit_toas(
        self,
        maxiter=20,
        required_chi2_decrease=1e-2,
        max_chi2_increase=1e-2,
        min_lambda=1e-3,
        debug=False,
    ):
        """Downhill fit implementation for fitting the timing model parameters.
        The `fit_toas()` calls this method iteratively to fit the timing model parameters
        while also fitting for white noise parameters.

        See documentation of the `fit_toas()` method for more details."""
        # setup
        self.model.validate()
        self.model.validate_toas(self.toas)
        current_state = self.create_state()
        best_state = current_state
        self.converged = False
        # algorithm
        exception = None

        for i in range(maxiter):
            step = current_state.step
            lambda_ = 1
            chi2_decrease = 0
            while True:
                try:
                    new_state = current_state.take_step(step, lambda_)
                    chi2_decrease = current_state.chi2 - new_state.chi2
                    if new_state.chi2 < best_state.chi2:
                        best_state = new_state
                    if chi2_decrease < -max_chi2_increase:
                        raise InvalidModelParameters(
                            f"chi2 increased from {current_state.chi2} to {new_state.chi2} "
                            f"when trying to take a step with lambda {lambda_}"
                        )
                    log.trace(
                        f"Iteration {i}: "
                        f"Updating state, chi2 goes down by {chi2_decrease} "
                        f"from {current_state.chi2} "
                        f"to {new_state.chi2}"
                    )
                    exception = None
                    current_state = new_state
                    break
                except InvalidModelParameters as e:
                    # This could be an exception evaluating new_state.chi2 or an increase in value
                    # If bad parameter values escape, look in ModelState.resids for the except
                    # that should catch them
                    lambda_ /= 2
                    log.trace(f"Iteration {i}: Shortening step to {lambda_}: {e}")
                    if lambda_ < min_lambda:
                        log.warning(
                            f"Unable to improve chi2 even with very small steps, stopping "
                            f"but keeping best state, message was: {e}"
                        )
                        exception = e
                        break
            if (
                -max_chi2_increase <= chi2_decrease < required_chi2_decrease
                and lambda_ == 1
            ):
                log.debug(
                    f"Iteration {i}: chi2 does not improve, stopping; "
                    f"decrease: {chi2_decrease}"
                )
                self.converged = True
                break
            if exception is not None:
                break
        else:
            log.debug(
                f"Stopping because maximum number of iterations ({maxiter}) reached"
            )

        self.current_state = best_state
        # collect results
        self.model = self.current_state.model
        self.resids = self.current_state.resids
        self.parameter_covariance_matrix = (
            self.current_state.parameter_covariance_matrix
        )
        self.errors = np.sqrt(np.diag(self.parameter_covariance_matrix.matrix))
        self.parameter_correlation_matrix = (
            self.parameter_covariance_matrix.to_correlation_matrix()
        )

        for p, e in zip(self.current_state.params, self.errors):
            try:
                # I don't know why this fails with multiprocessing, but bypass if it does
                with contextlib.suppress(ValueError):
                    log.trace(f"Setting {getattr(self.model, p)} uncertainty to {e}")
                pm = getattr(self.model, p)
            except AttributeError:
                if p != "Offset":
                    log.warning(f"Unexpected parameter {p}")
            else:
                pm.uncertainty = e * pm.units
        self.update_model(self.current_state.chi2)
        if exception is not None:
            raise StepProblem(
                "Unable to improve chi2 even with very small steps"
            ) from exception
        if not self.converged:
            raise MaxiterReached(f"Convergence not detected after {maxiter} steps.")
        return self.converged

    def fit_toas(
        self,
        maxiter=20,
        noise_fit_niter=2,
        required_chi2_decrease=1e-2,
        max_chi2_increase=1e-2,
        min_lambda=1e-3,
        noisefit_method="Newton-CG",
        compute_noise_uncertainties=True,
        debug=False,
    ):
        """Carry out a cautious downhill fit.

        This tries to take the same steps as
        :func:`pint.fitter.WLSFitter.fit_toas` or
        :func:`pint.fitter.GLSFitter.fit_toas` or
        :func:`pint.fitter.WidebandTOAFitter.fit_toas`.  At each step, it
        checks whether the new model has a better ``chi2`` than the current
        one; if the new model is invalid or worse than the current one, it
        tries taking a shorter step in the same direction. This can exit if it
        exceeds the maximum number of iterations or if improvement is not
        possible even with very short steps, or it can exit successfully if a
        full-size step is taken and it does not decrease the ``chi2`` by much.

        The attribute ``self.converged`` is set to True or False depending on
        whether the process actually converged.

        This function can also estimate white noise parameters (EFACs and EQUADs)
        and their uncertainties.

        If there are no free white noise parameters, this function will do one
        iteration of the downhill fit (implemented in the `_fit_toas()` method).
        If free white noise parameters are present, it will fit for them by numerically
        maximizing the likelihood function (implemented in the `_fit_noise()` method).
        The timing model fit and the noise model fit are run iteratively in an alternating
        fashion. Fitting for a white noise parameter is as simple as::

            fitter.model.EFAC1.frozen = False
            fitter.fit_toas()


        Parameters
        ==========

        maxiter : int
            Abandon the process if this many successful steps have been taken.
        required_chi2_decrease : float
            A full-size step that makes less than this much improvement is taken
            to indicate that the fitter has converged.
        max_chi2_increase : float
            If this is positive, consider taking steps that slightly worsen the chi2 in hopes
            of eventually finding our way downhill.
        min_lambda : float
            If steps are shrunk by this factor and still don't result in improvement, abandon hope
            of convergence and stop.
        noisefit_method: str
            Algorithm used to fit for noise parameters. See the documentation for
            `scipy.optimize.minimize()` for more details and available options.
        """
        free_noise_params = self._get_free_noise_params()

        if len(free_noise_params) == 0:
            return self._fit_toas(
                maxiter=maxiter,
                required_chi2_decrease=required_chi2_decrease,
                max_chi2_increase=required_chi2_decrease,
                min_lambda=required_chi2_decrease,
                debug=debug,
            )

        log.debug("Will fit for noise parameters.")
        for ii in range(noise_fit_niter):
            self._fit_toas(
                maxiter=maxiter,
                required_chi2_decrease=required_chi2_decrease,
                max_chi2_increase=max_chi2_increase,
                min_lambda=min_lambda,
                debug=debug,
            )

            if ii == noise_fit_niter - 1 and compute_noise_uncertainties:
                values, errors = self._fit_noise(
                    noisefit_method=noisefit_method, uncertainty=True
                )
                self._update_noise_params(values, errors)
            else:
                values = self._fit_noise(
                    noisefit_method=noisefit_method, uncertainty=False
                )
                self._update_noise_params(values)

        return self._fit_toas(
            maxiter=maxiter,
            required_chi2_decrease=required_chi2_decrease,
            max_chi2_increase=max_chi2_increase,
            min_lambda=min_lambda,
            debug=debug,
        )

    @property
    def fac(self):
        return self.current_state.fac

    def _get_free_noise_params(self):
        """Returns a list of all free noise parameters."""
        return [
            fp
            for fp in self.model.get_params_of_component_type("NoiseComponent")
            if not getattr(self.model, fp).frozen
        ]

    def _update_noise_params(self, values, errors=None):
        """Update the model using estimated noise parameters."""
        free_noise_params = self._get_free_noise_params()

        if errors is not None:
            for fp, val, err in zip(free_noise_params, values, errors):
                getattr(self.model, fp).value = val
                getattr(self.model, fp).uncertainty_value = err
        else:
            for fp, val in zip(free_noise_params, values):
                getattr(self.model, fp).value = val

    def _fit_noise(self, noisefit_method="Newton-CG", uncertainty=False):
        """Estimate noise parameters and their uncertainties. Noise parameters
        are estimated by numerically maximizing the log-likelihood function including
        the normalization term. The uncertainties thereof are computed using the
        numerically-evaluated Hessian."""
        free_noise_params = self._get_free_noise_params()

        xs0 = [getattr(self.model, fp).value for fp in free_noise_params]

        model1 = copy.deepcopy(self.model)
        res = Residuals(self.toas, model1)

        def _mloglike(xs):
            """Negative of the log-likelihood function."""
            for fp, x in zip(free_noise_params, xs):
                getattr(res.model, fp).value = x

            return -res.lnlikelihood()

        if not res.model.has_correlated_errors:

            def _mloglike_grad(xs):
                """Gradient of the negative of the log-likelihood function w.r.t. white noise parameters."""
                for fp, x in zip(free_noise_params, xs):
                    getattr(res.model, fp).value = x

                return np.array(
                    [
                        -res.d_lnlikelihood_d_param(par).value
                        for par in free_noise_params
                    ]
                )

            maxlike_result = opt.minimize(
                _mloglike, xs0, method=noisefit_method, jac=_mloglike_grad
            )
        else:
            maxlike_result = opt.minimize(_mloglike, xs0, method="Nelder-Mead")

        if uncertainty:
            hess = Hessian(_mloglike)
            errs = np.sqrt(np.diag(np.linalg.pinv(hess(maxlike_result.x))))

        return (maxlike_result.x, errs) if uncertainty else maxlike_result.x


class WLSState(ModelState):
    def __init__(self, fitter, model, threshold=None):
        super().__init__(fitter, model)
        self.threshold = threshold

    @cached_property
    def step(self):
        # Define the linear system
        M, params, units = self.model.designmatrix(
            toas=self.fitter.toas, incfrozen=False, incoffset=True
        )
        # Get residuals and TOA uncertainties in seconds
        Nvec = self.model.scaled_toa_uncertainty(self.fitter.toas).to(u.s).value
        scaled_resids = self.resids.time_resids.to(u.s).value / Nvec

        # "Whiten" design matrix and residuals by dividing by uncertainties
        M = M / Nvec.reshape((-1, 1))

        # For each column in design matrix except for col 0 (const. pulse
        # phase), subtract the mean value, and scale by the column RMS.
        # This helps avoid numerical problems later.  The scaling factors need
        # to be saved to recover correct parameter units.
        # NOTE, We remove subtract mean value here, since it did not give us a
        # fast converge fitting.
        # M[:,1:] -= M[:,1:].mean(axis=0)
        M, fac = normalize_designmatrix(M, params)
        # Singular value decomp of design matrix:
        #   M = U s V^T
        # Dimensions:
        #   M, U are Ntoa x Nparam
        #   s is Nparam x Nparam diagonal matrix encoded as 1-D vector
        #   V^T is Nparam x Nparam
        U, s, Vt = scipy.linalg.svd(M, full_matrices=False)
        # Note, here we could do various checks like report
        # matrix condition number or zero out low singular values.
        # print 'log_10 cond=', np.log10(s.max()/s.min())
        # Note, Check the threshold from data precision level.Borrowed from
        # np Curve fit.
        threshold = self.threshold
        if threshold is None:
            # M is float, not longdouble
            # threshold = np.finfo(float).eps * max(M.shape)
            threshold = 1e-14 * max(M.shape)

        log.trace(f"Singular values for fit are {s}")
        bad = np.where(s <= threshold * s[0])[0]
        s[bad] = np.inf
        for c in bad:
            bad_col = Vt[c, :]
            bad_col /= abs(bad_col).max()
            bad_combination = " + ".join(
                [
                    f"{co}*{p}"
                    for (co, p) in sorted(zip(bad_col, params))
                    if abs(co) > threshold
                ]
            )
            warn(
                f"Parameter degeneracy; the following linear combination yields "
                f"almost no change: {bad_combination}",
                DegeneracyWarning,
            )

        self.M = M
        self.U = U
        self.Vt = Vt
        self.s = s
        self.fac = fac
        self.params = params
        self.units = units
        self.scaled_resids = scaled_resids
        # TODO: seems like doing this on every iteration is wasteful, and we should just do it once and then update the matrix
        covariance_matrix_labels = {
            param: (i, i + 1, unit)
            for i, (param, unit) in enumerate(zip(params, units))
        }
        # covariance matrix is 2D and symmetric
        covariance_matrix_labels = [covariance_matrix_labels] * 2
        self.parameter_covariance_matrix_labels = covariance_matrix_labels

        # The delta-parameter values
        #   dpars = V s^-1 U^T r
        # Scaling by fac recovers original units
        return (Vt.T @ ((U.T @ scaled_resids) / s)) / fac

    def take_step(self, step, lambda_=1):
        return WLSState(
            self.fitter, self.take_step_model(step, lambda_), threshold=self.threshold
        )

    @cached_property
    def parameter_covariance_matrix(self):
        # make sure we compute the SVD
        self.step
        # Sigma = np.dot(Vt.T / s, U.T)
        # The post-fit parameter covariance matrix
        #   Sigma = V s^-2 V^T
        Sigma = np.dot(self.Vt.T / (self.s**2), self.Vt)
        return CovarianceMatrix(
            (Sigma / self.fac).T / self.fac, self.parameter_covariance_matrix_labels
        )


class DownhillWLSFitter(DownhillFitter):
    """Fitter that uses the shortening-step procedure for WLS fits.

    Most of the machinery here is in :class:`pint.fitter.WLSState`
    or :class:`pint.fitter.DownhillFitter`.
    """

    def __init__(self, toas, model, track_mode=None, residuals=None):
        if model.has_correlated_errors:
            raise CorrelatedErrors(model)
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.method = "downhill_wls"

    def fit_toas(self, maxiter=10, threshold=None, debug=False, **kwargs):
        """Fit TOAs.

        This is mostly implemented in
        :func:`pint.fitter.DownhillFitter.fit_toas`.

        Parameters
        ==========
        maxiter : int
            Abandon hope if convergence hasn't occurred after this many steps (successful or not).
        threshold : float
            Discard singular values less than this times the largest; this makes the linear algebra
            a little more stable, but the Levenberg-Marquardt algorithm is supposed to do that anyway.
        kwargs : dict
            Any additional arguments are passed down to
            :func:`pint.fitter.DownhillFitter.fit_toas`
        """
        self.threshold = threshold
        super().fit_toas(maxiter=maxiter, debug=debug, **kwargs)

    def create_state(self):
        return WLSState(self, self.model)


class GLSState(ModelState):
    def __init__(self, fitter, model, full_cov=False, threshold=None):
        super().__init__(fitter, model)
        self.threshold = threshold
        self.full_cov = full_cov

    @cached_property
    def step(self):
        # Define the linear system
        M, params, units = self.model.designmatrix(
            toas=self.fitter.toas, incfrozen=False, incoffset=True
        )
        self.params = params
        self.units = units
        # TODO: seems like doing this on every iteration is wasteful, and we should just do it once and then update the matrix
        covariance_matrix_labels = {
            param: (i, i + 1, unit)
            for i, (param, unit) in enumerate(zip(params, units))
        }
        # covariance matrix is 2D and symmetric
        covariance_matrix_labels = [covariance_matrix_labels] * 2
        self.parameter_covariance_matrix_labels = covariance_matrix_labels

        residuals = self.resids.time_resids.to(u.s).value

        # get any noise design matrices and weight vectors
        if not self.full_cov:
            Mn = self.model.noise_model_designmatrix(self.fitter.toas)
            phi = self.model.noise_model_basis_weight(self.fitter.toas)
            phiinv = np.zeros(M.shape[1])
            if Mn is not None and phi is not None:
                phiinv = np.concatenate((phiinv, 1 / phi))
                M = np.hstack((M, Mn))

        # normalize the design matrix
        M, norm = normalize_designmatrix(M, params)
        self.M = M
        self.fac = norm

        # compute covariance matrices
        if self.full_cov:
            cov = self.model.toa_covariance_matrix(self.fitter.toas)
            cf = scipy.linalg.cho_factor(cov)
            cm = scipy.linalg.cho_solve(cf, M)
            mtcm = np.dot(M.T, cm)
            mtcy = np.dot(cm.T, residuals)

        else:
            phiinv /= norm**2
            # Why are we scaling residuals by the *square* of the uncertainty?
            Nvec = (
                self.model.scaled_toa_uncertainty(self.fitter.toas).to(u.s).value ** 2
            )
            cinv = 1 / Nvec
            mtcm = np.dot(M.T, cinv[:, None] * M)
            mtcm += np.diag(phiinv)
            mtcy = np.dot(M.T, cinv * residuals)
        log.trace(f"mtcm: {mtcm}")

        U, s, Vt = scipy.linalg.svd(mtcm, full_matrices=False)
        log.trace(f"s: {s}")

        bad = np.where(s <= self.threshold * s[0])[0]
        s[bad] = np.inf
        for c in bad:
            bad_col = Vt[c, :]
            bad_col /= abs(bad_col).max()
            bad_combination = " ".join(
                [
                    f"{p}"
                    for (co, p) in sorted(zip(bad_col, params))
                    if abs(co) > self.threshold
                ]
            )
            warn(
                f"Parameter degeneracy; the following combination of parameters yields "
                f"almost no change: {bad_combination}",
                DegeneracyWarning,
            )

        self.norm = norm
        self.s, self.Vt = s, Vt
        xhat = np.dot(Vt.T, np.dot(U.T, mtcy) / s)
        log.trace(f"norm: {norm}")
        log.trace(f"xhat: {xhat}")
        self.xhat = xhat
        # newres = residuals - np.dot(M, xhat)

        # compute absolute estimates, normalized errors, covariance matrix
        return xhat / norm

    def take_step(self, step, lambda_=1):
        return GLSState(
            self.fitter,
            self.take_step_model(step, lambda_),
            threshold=self.threshold,
            full_cov=self.full_cov,
        )

    @cached_property
    def parameter_covariance_matrix(self):
        # make sure we compute the SVD
        self.step
        xvar = np.dot(self.Vt.T / self.s, self.Vt)
        return CovarianceMatrix(
            (xvar / self.norm).T / self.norm, self.parameter_covariance_matrix_labels
        )


class DownhillGLSFitter(DownhillFitter):
    """Fitter that uses the shortening-step procedure for GLS fits.

    Most of the machinery here is in :class:`pint.fitter.GLSState`
    or :class:`pint.fitter.DownhillFitter`.
    """

    # FIXME: do something clever to efficiently compute chi-squared

    def __init__(self, toas, model, track_mode=None, residuals=None):
        if not model.has_correlated_errors:
            log.info(
                "Model does not appear to have correlated errors so the GLS fitter "
                "is unnecessary; DownhillWLSFitter may be faster and more stable."
            )
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.method = "downhill_gls"
        self.full_cov = False
        self.threshold = 0

    def create_state(self):
        return GLSState(
            self, self.model, full_cov=self.full_cov, threshold=self.threshold
        )

    def fit_toas(self, maxiter=10, threshold=0, full_cov=False, debug=False, **kwargs):
        """Fit TOAs.

        This is mostly implemented in
        :func:`pint.fitter.DownhillFitter.fit_toas`.

        Parameters
        ==========
        maxiter : int
            Abandon hope if convergence hasn't occurred after this many steps (successful or not).
        threshold : float
            Discard singular values less than this times the largest; this makes the linear algebra
            a little more stable, but the Levenberg-Marquardt algorithm is supposed to do that anyway.
        full_cov : bool
            If True, use the full TOA covariance matrix, which can be huge; if False, use the
            rank-reduced approach (for which Levenberg-Marquardt may not make sense).
        kwargs : dict
            Any additional arguments are passed down to
            :func:`pint.fitter.DownhillFitter.fit_toas`
        """
        self.threshold = threshold
        self.full_cov = full_cov
        r = super().fit_toas(maxiter=maxiter, debug=debug, **kwargs)

        # FIXME: set up noise residuals et cetera
        # Compute the noise realizations if possible
        if not self.full_cov:
            noise_dims = self.model.noise_model_dimensions(self.toas)
            noise_resids = {}
            ntmpar = len(self.model.free_params)
            for comp in noise_dims:
                # The first column of designmatrix is "offset", add 1 to match
                # the indices of noise designmatrix
                p0 = noise_dims[comp][0] + ntmpar + 1
                p1 = p0 + noise_dims[comp][1]
                noise_resids[comp] = (
                    np.dot(
                        self.current_state.M[:, p0:p1], self.current_state.xhat[p0:p1]
                    )
                    * u.s
                )
                if debug:
                    setattr(
                        self.resids,
                        f"{comp}_M",
                        (
                            self.current_state.M[:, p0:p1],
                            self.current_state.xhat[p0:p1],
                        ),
                    )
                    setattr(self.resids, f"{comp}_M_index", (p0, p1))
            self.resids.noise_resids = noise_resids
            if debug:
                setattr(self.resids, "norm", self.current_state.norm)

        return r


class WidebandState(ModelState):
    def __init__(self, fitter, model, full_cov=False, threshold=None):
        super().__init__(fitter, model)
        self.threshold = threshold
        self.full_cov = full_cov
        self.add_args = {}  # for adding arguments to residual creation

    @cached_property
    def M_params_units_norm(self):
        # Define the linear system
        d_matrix = combine_design_matrices_by_quantity(
            [
                DesignMatrixMaker("toa", u.s)(
                    self.fitter.toas, self.model, self.model.free_params, offset=True
                ),
                DesignMatrixMaker("dm", u.pc / u.cm**3)(
                    self.fitter.toas, self.model, self.model.free_params, offset=True
                ),
            ]
        )
        M, params, units = (
            d_matrix.matrix,
            d_matrix.derivative_params,
            d_matrix.param_units,
        )
        # get any noise design matrices and weight vectors
        if not self.full_cov:
            # We assume the fit date type is toa
            Mn = DesignMatrixMaker("toa_noise", u.s)(self.fitter.toas, self.model)
            phi = self.model.noise_model_basis_weight(self.fitter.toas)
            phiinv = np.zeros(M.shape[1])
            if Mn is not None and phi is not None:
                phiinv = np.concatenate((phiinv, 1 / phi))
                new_d_matrix = combine_design_matrices_by_param(d_matrix, Mn)
                M, params, units = (
                    new_d_matrix.matrix,
                    new_d_matrix.derivative_params,
                    new_d_matrix.param_units,
                )

        # normalize the design matrix
        norm = np.sqrt(np.sum(M**2, axis=0))
        # The fixed offset is an unlisted parameter
        ntmpar = len(self.model.free_params) + 1
        if M.shape[1] > ntmpar:
            norm[ntmpar:] = 1
        for c in np.where(norm == 0)[0]:
            warn(
                f"Parameter degeneracy; the following parameter yields "
                f"almost no change: {params[c]}",
                DegeneracyWarning,
            )
        norm[norm == 0] = 1
        M /= norm
        if not self.full_cov:
            phiinv /= norm**2
            self.phiinv = phiinv
        self.fac = norm

        return M, params, units, norm

    @cached_property
    def M(self):
        return self.M_params_units_norm[0]

    @cached_property
    def params(self):
        return self.M_params_units_norm[1]

    @cached_property
    def units(self):
        return self.M_params_units_norm[2]

    @cached_property
    def norm(self):
        return self.M_params_units_norm[3]

    @cached_property
    def mtcm_mtcy_mtcmplain(self):
        # FIXME: ensure that TOAs are before DM
        residuals = np.hstack(
            (self.resids.toa.time_resids.to_value(u.s), self.resids.dm.resids_value)
        )

        # compute covariance matrices
        if self.full_cov:
            cov = combine_covariance_matrix(
                [
                    CovarianceMatrixMaker("toa", u.s)(self.fitter.toas, self.model),
                    CovarianceMatrixMaker("dm", u.pc / u.cm**3)(
                        self.fitter.toas, self.model
                    ),
                ]
            ).matrix
            cf = scipy.linalg.cho_factor(cov)
            cm = scipy.linalg.cho_solve(cf, self.M)
            mtcm = np.dot(self.M.T, cm)
            mtcy = np.dot(cm.T, residuals)
            mtcmplain = mtcm
        else:
            Nvec = (
                np.hstack(
                    [
                        self.model.scaled_toa_uncertainty(self.fitter.toas).to_value(
                            u.s
                        ),
                        self.model.scaled_dm_uncertainty(self.fitter.toas).to_value(
                            u.pc / u.cm**3
                        ),
                    ]
                )
                ** 2
            )
            cinv = 1 / Nvec
            mtcm = np.dot(self.M.T, cinv[:, None] * self.M)
            mtcmplain = mtcm
            mtcm += np.diag(self.phiinv)
            mtcy = np.dot(self.M.T, cinv * residuals)
        return mtcm, mtcy, mtcmplain

    @cached_property
    def mtcm(self):
        return self.mtcm_mtcy_mtcmplain[0]

    @cached_property
    def mtcy(self):
        return self.mtcm_mtcy_mtcmplain[1]

    @cached_property
    def mtcmplain(self):
        return self.mtcm_mtcy_mtcmplain[2]

    @cached_property
    def U_s_Vt_xhat(self):
        U, s, Vt = scipy.linalg.svd(self.mtcm, full_matrices=False)

        bad = np.where(s <= self.threshold * s[0])[0]
        s[bad] = np.inf
        for c in bad:
            bad_col = Vt[c, :]
            bad_col /= abs(bad_col).max()
            bad_combination = " ".join(
                [
                    f"{co}*{p}"
                    for (co, p) in reversed(sorted(zip(bad_col, self.params)))
                    if abs(co) > self.threshold
                ]
            )
            warn(
                f"Parameter degeneracy; the following combination of parameters yields "
                f"almost no change: {bad_combination}",
                DegeneracyWarning,
            )

        xhat = np.dot(Vt.T, np.dot(U.T, self.mtcy) / s)
        return U, s, Vt, xhat

    @cached_property
    def U(self):
        return self.U_s_Vt_xhat[0]

    @cached_property
    def s(self):
        return self.U_s_Vt_xhat[1]

    @cached_property
    def Vt(self):
        return self.U_s_Vt_xhat[2]

    @cached_property
    def xhat(self):
        return self.U_s_Vt_xhat[3]

    @cached_property
    def step(self):
        # compute absolute estimates, normalized errors, covariance matrix
        return self.xhat / self.norm

    def take_step(self, step, lambda_=1):
        return WidebandState(
            self.fitter, self.take_step_model(step, lambda_), threshold=self.threshold
        )

    @cached_property
    def parameter_covariance_matrix(self):
        # make sure we compute the SVD
        xvar = np.dot(self.Vt.T / self.s, self.Vt)
        # is this the best place to do this?
        covariance_matrix_labels = {
            param: (i, i + 1, unit)
            for i, (param, unit) in enumerate(zip(self.params, self.units))
        }
        # covariance matrix is 2D and symmetric
        covariance_matrix_labels = [covariance_matrix_labels] * 2

        return CovarianceMatrix(
            (xvar / self.norm).T / self.norm, covariance_matrix_labels
        )


class WidebandDownhillFitter(DownhillFitter):
    """Fitter that uses the shortening-step procedure for wideband GLS fits.

    Most of the machinery here is in :class:`pint.fitter.WidebandState`
    or :class:`pint.fitter.DownhillFitter`.
    """

    # FIXME: do something clever to efficiently compute chi-squared

    def __init__(self, toas, model, track_mode=None, residuals=None, add_args=None):
        self.method = "downhill_wideband"
        self.full_cov = False
        self.threshold = 0
        self.add_args = {} if add_args is None else add_args
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.is_wideband = True

    def make_resids(self, model):
        return WidebandTOAResiduals(
            self.toas,
            model,
            toa_resid_args=self.add_args.get("toa", {}),
            dm_resid_args=self.add_args.get("dm", {}),
        )

    def create_state(self):
        return WidebandState(
            self, self.model, full_cov=self.full_cov, threshold=self.threshold
        )

    def fit_toas(
        self, maxiter=10, threshold=1e-14, full_cov=False, debug=False, **kwargs
    ):
        """Fit TOAs.

        This is mostly implemented in
        :func:`pint.fitter.DownhillFitter.fit_toas`.

        Parameters
        ==========
        maxiter : int
            Abandon hope if convergence hasn't occurred after this many steps (successful or not).
        threshold : float
            Discard singular values less than this times the largest; this makes the linear algebra
            a little more stable, but the Levenberg-Marquardt algorithm is supposed to do that anyway.
        full_cov : bool
            If True, use the full TOA covariance matrix, which can be huge; if False, use the
            rank-reduced approach (for which Levenberg-Marquardt may not make sense).
        kwargs : dict
            Any additional arguments are passed down to
            :func:`pint.fitter.DownhillFitter.fit_toas`
        """
        self.threshold = threshold
        self.full_cov = full_cov
        # FIXME: set up noise residuals et cetera
        r = super().fit_toas(maxiter=maxiter, debug=debug, **kwargs)
        # Compute the noise realizations if possibl
        if not self.full_cov:
            noise_dims = self.model.noise_model_dimensions(self.toas)
            noise_resids = {}
            ntmpar = len(self.model.free_params)
            for comp in noise_dims:
                # The first column of designmatrix is "offset", add 1 to match
                # the indices of noise designmatrix
                p0 = noise_dims[comp][0] + ntmpar + 1
                p1 = p0 + noise_dims[comp][1]
                noise_resids[comp] = (
                    np.dot(
                        self.current_state.M[:, p0:p1], self.current_state.xhat[p0:p1]
                    )
                    * u.s
                )
                if debug:
                    setattr(
                        self.resids,
                        f"{comp}_M",
                        (
                            self.current_state.M[:, p0:p1],
                            self.current_state.xhat[p0:p1],
                        ),
                    )
                    setattr(self.resids, f"{comp}_M_index", (p0, p1))
            self.resids.noise_resids = noise_resids
            if debug:
                setattr(self.resids, "norm", self.current_state.norm)
        return r


class PowellFitter(Fitter):
    """A fitter that demonstrates how to work with a generic fitting function.

    This class wraps ``scipy.optimize.minimize`` with ``method="Powell"``; this
    uses Powell's method, a simplex optimizer that makes no use of derivatives
    (and is thus rather slow by comparison).

    It is unlikely that this will be useful for timing an actual pulsar, but it
    may serve as an example of how to write your own fitting procedure.
    """

    def __init__(self, toas, model, track_mode=None, residuals=None):
        super().__init__(toas, model, residuals=residuals, track_mode=track_mode)
        self.method = "Powell"

    def fit_toas(self, maxiter=20, debug=False):
        """Carry out the fitting procedure."""
        # check that params of timing model have necessary components
        self.model.validate()
        self.model.validate_toas(self.toas)
        # Initial guesses are model params
        fitp = self.model.get_params_dict("free", "num")
        self.fitresult = opt.minimize(
            self.minimize_func,
            list(fitp.values()),
            args=tuple(fitp.keys()),
            options={"maxiter": maxiter},
            method=self.method,
        )
        # Update model and resids, as the last iteration of minimize is not
        # necessarily the one that yields the best fit
        self.minimize_func(np.atleast_1d(self.fitresult.x), *list(fitp.keys()))

        self.update_model(self.resids.chi2)

        return self.resids.chi2


class WLSFitter(Fitter):
    """Weighted Least Squares fitter.

    This is the primary fitting algorithm in TEMPO and TEMPO2 as well as PINT.
    It requires derivatives and it cannot handle correlated errors (ECORR, for
    example).

    This fitter implements a rudimentary form of the Gauss-Newton algorithm: it
    computes an initial set of residuals from the fit, it computes the design
    matrix, and it treats the problem as a weighted linear least squares
    problem, jumping immediately to the assumed solution. If the initial guess
    is sufficiently close that the objective function is well-approximated by
    its derivatives, this lands very close to the correct answer. This Fitter
    can be told to repeat the process a number of times. The
    Levenburg-Marquardt algorithm and its descendants (trust region algorithms)
    generalize this process to handle situations where the initial guess is not
    close enough that the derivatives are a good approximation.
    """

    def __init__(self, toas, model, track_mode=None, residuals=None):
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.method = "weighted_least_square"

    def fit_toas(self, maxiter=1, threshold=None, debug=False):
        """Run a linear weighted least-squared fitting method.

        Parameters
        ----------
        maxiter: int
            Repeat the least-squares fitting up to this many times if necessary.
        threshold : float or None
            Discard singular values smaller than ``threshold`` times the largest
            singular value. If None, use a value based on floating-point epsilon
            and the matrix sizes.
        """
        # check that params of timing model have necessary components
        self.model.validate()
        self.model.validate_toas(self.toas)
        chi2 = 0
        for _ in range(maxiter):
            fitp = self.model.get_params_dict("free", "quantity")
            fitpv = self.model.get_params_dict("free", "num")
            fitperrs = self.model.get_params_dict("free", "uncertainty")
            # Define the linear system
            M, params, units = self.get_designmatrix()
            # Get residuals and TOA uncertainties in seconds
            self.update_resids()
            residuals = self.resids.time_resids.to(u.s).value
            Nvec = self.model.scaled_toa_uncertainty(self.toas).to(u.s).value

            # "Whiten" design matrix and residuals by dividing by uncertainties
            M = M / Nvec.reshape((-1, 1))
            residuals = residuals / Nvec

            # For each column in design matrix except for col 0 (const. pulse
            # phase), subtract the mean value, and scale by the column RMS.
            # This helps avoid numerical problems later.  The scaling factors need
            # to be saved to recover correct parameter units.
            # NOTE, We remove subtract mean value here, since it did not give us a
            # fast converge fitting.
            # M[:,1:] -= M[:,1:].mean(axis=0)
            M, fac = normalize_designmatrix(M, params)
            # Singular value decomp of design matrix:
            #   M = U s V^T
            # Dimensions:
            #   M, U are Ntoa x Nparam
            #   s is Nparam x Nparam diagonal matrix encoded as 1-D vector
            #   V^T is Nparam x Nparam
            U, s, Vt = scipy.linalg.svd(M, full_matrices=False)
            # Note, here we could do various checks like report
            # matrix condition number or zero out low singular values.
            # print 'log_10 cond=', np.log10(s.max()/s.min())
            # Note, Check the threshold from data precision level.Borrowed from
            # np Curve fit.
            if threshold is None:
                # M is float, not longdouble
                # threshold = np.finfo(float).eps * max(M.shape)
                threshold = 1e-14 * max(M.shape)

            bad = np.where(s <= threshold * s[0])[0]
            s[bad] = np.inf
            for c in bad:
                bad_col = Vt[c, :]
                bad_col /= abs(bad_col).max()
                bad_combination = " + ".join(
                    [
                        f"{co}*{p}"
                        for (co, p) in reversed(sorted(zip(bad_col, params)))
                        if abs(co) > threshold
                    ]
                )
                warn(
                    f"Parameter degeneracy; the following linear combination yields "
                    f"almost no change: {bad_combination}",
                    DegeneracyWarning,
                )
            # Sigma = np.dot(Vt.T / s, U.T)
            # The post-fit parameter covariance matrix
            #   Sigma = V s^-2 V^T
            Sigma = np.dot(Vt.T / (s**2), Vt)
            # Parameter uncertainties. Scale by fac recovers original units.
            errs = np.sqrt(np.diag(Sigma)) / fac
            # covariance matrix stuff (for randomized models in pintk)
            sigma_var = (Sigma / fac).T / fac
            errors = np.sqrt(np.diag(sigma_var))
            sigma_cov = (sigma_var / errors).T / errors
            # covariance matrix = variances in diagonal, used for gaussian random models
            covariance_matrix = sigma_var
            covariance_matrix_labels = {
                param: (i, i + 1, unit)
                for i, (param, unit) in enumerate(zip(params, units))
            }
            # covariance matrix is 2D and symmetric
            covariance_matrix_labels = [
                covariance_matrix_labels
            ] * covariance_matrix.ndim
            self.parameter_covariance_matrix = CovarianceMatrix(
                covariance_matrix, covariance_matrix_labels
            )

            # correlation matrix = 1s in diagonal, use for comparison to tempo/tempo2 cov matrix
            self.parameter_correlation_matrix = CorrelationMatrix(
                sigma_cov, covariance_matrix_labels
            )
            self.fac = fac
            self.errors = errors

            # The delta-parameter values
            #   dpars = V s^-1 U^T r
            # Scaling by fac recovers original units
            dpars = np.dot(Vt.T, np.dot(U.T, residuals) / s) / fac
            for pn in fitp.keys():
                uind = params.index(pn)  # Index of designmatrix
                un = 1.0 / (units[uind])  # Unit in designmatrix
                un *= u.s
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
                # NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            chi2 = self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)

        self.update_model(chi2)

        return chi2


class GLSFitter(Fitter):
    """Generalized least-squares fitting.

    This fitter extends the :class:`pint.fitter.WLSFitter` to permit the errors
    on the data points to be correlated. The fitting proceeds by decomposing
    the data covariance matrix.
    """

    def __init__(self, toas=None, model=None, track_mode=None, residuals=None):
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.method = "generalized_least_square"

    def fit_toas(self, maxiter=1, threshold=0, full_cov=False, debug=False):
        """Run a generalized least-squares fitting method.

        A first attempt is made to solve the fitting problem by Cholesky
        decomposition, but if this fails singular value decomposition is
        used instead. In this case singular values below threshold are removed.

        Parameters
        ----------
        maxiter: int
            How many times to run the linear least-squares fit, re-evaluating
            the derivatives at each step.
            If maxiter is less than one, no fitting is done, just the
            chi-squared computation. In this case, you must provide the residuals
            argument when constructing the class.
            If maxiter is one or more, so fitting is actually done, the
            chi-squared value returned is only approximately the chi-squared
            of the improved(?) model. In fact it is the chi-squared of the
            solution to the linear fitting problem, and the full non-linear
            model should be evaluated and new residuals produced if an accurate
            chi-squared is desired.
        threshold: float
            When to start discarding singular values. Typical values are about
            1e-14 - a singular value smaller than this indicates parameters that
            are so degenerate the numerical precision cannot distinguish them.
            Such highly degenerate parameter sets are reported to the user with
            a DegeneracyWarning.  Negative values force a faster but less stable
            Cholesky decomposition method.
        full_cov: bool
            full_cov determines which calculation is used. If true, the full
            covariance matrix is constructed and the calculation is relatively
            straightforward but the full covariance matrix may be enormous.
            If false, an algorithm is used that takes advantage of the structure
            of the covariance matrix, based on information provided by the noise
            model. The two algorithms should give the same result to numerical
            accuracy where they both can be applied.
        """
        # check that params of timing model have necessary components
        self.model.validate()
        self.model.validate_toas(self.toas)
        chi2 = 0
        for i in range(maxiter):
            fitp = self.model.get_params_dict("free", "quantity")
            fitpv = self.model.get_params_dict("free", "num")
            fitperrs = self.model.get_params_dict("free", "uncertainty")

            # Define the linear system
            # normalize the design matrix
            M, params, units = self.get_designmatrix()
            # M /= norm

            ntmpar = len(fitp)

            # Get residuals and TOA uncertainties in seconds
            if i == 0:
                # Why is this here?
                self.update_resids()
            residuals = self.resids.time_resids.to(u.s).value

            # get any noise design matrices and weight vectors
            if not full_cov:
                Mn = self.model.noise_model_designmatrix(self.toas)
                phi = self.model.noise_model_basis_weight(self.toas)
                phiinv = np.zeros(M.shape[1])
                if Mn is not None and phi is not None:
                    phiinv = np.concatenate((phiinv, 1 / phi))
                    M = np.hstack((M, Mn))

            ntmpar = len(fitp)

            # normalize the design matrix
            M, norm = normalize_designmatrix(M, params)
            self.fac = norm

            # compute covariance matrices
            if full_cov:
                cov = self.model.toa_covariance_matrix(self.toas)
                cf = scipy.linalg.cho_factor(cov)
                cm = scipy.linalg.cho_solve(cf, M)
                mtcm = np.dot(M.T, cm)
                mtcy = np.dot(cm.T, residuals)

            else:
                phiinv /= norm**2
                Nvec = self.model.scaled_toa_uncertainty(self.toas).to(u.s).value ** 2
                cinv = 1 / Nvec
                mtcm = np.dot(M.T, cinv[:, None] * M)
                mtcm += np.diag(phiinv)
                mtcy = np.dot(M.T, cinv * residuals)

            log.trace(f"mtcm: {mtcm}")
            xhat, xvar = None, None
            if threshold <= 0:
                try:
                    c = scipy.linalg.cho_factor(mtcm)
                    xhat = scipy.linalg.cho_solve(c, mtcy)
                    xvar = scipy.linalg.cho_solve(c, np.eye(len(mtcy)))
                except scipy.linalg.LinAlgError:
                    xhat, xvar = None, None
            if xhat is None:
                U, s, Vt = scipy.linalg.svd(mtcm, full_matrices=False)
                log.trace(f"s: {s}")

                bad = np.where(s <= threshold * s[0])[0]
                s[bad] = np.inf
                for c in bad:
                    bad_col = Vt[c, :]
                    bad_col /= abs(bad_col).max()
                    bad_combination = " ".join(
                        [
                            f"{co}*{p}"
                            for (co, p) in reversed(sorted(zip(bad_col, params)))
                            if abs(co) > threshold
                        ]
                    )
                    warn(
                        f"Parameter degeneracy; the following combination of parameters yields "
                        f"almost no change: {bad_combination}",
                        DegeneracyWarning,
                    )

                xvar = np.dot(Vt.T / s, Vt)
                xhat = np.dot(Vt.T, np.dot(U.T, mtcy) / s)
            log.trace(f"norm: {norm}")
            log.trace(f"xhat: {xhat}")
            newres = residuals - np.dot(M, xhat)

            # compute linearized chisq
            # if full_cov:
            #     chi2 = np.dot(newres, scipy.linalg.cho_solve(cf, newres))
            # else:
            #     chi2 = np.dot(newres, cinv * newres) + np.dot(xhat, phiinv * xhat)

            # compute absolute estimates, normalized errors, covariance matrix
            dpars = xhat / norm
            errs = np.sqrt(np.diag(xvar)) / norm
            covmat = (xvar / norm).T / norm
            covariance_matrix_labels = {
                param: (i, i + 1, unit)
                for i, (param, unit) in enumerate(zip(params, units))
            }
            # covariance matrix is 2D and symmetric
            covariance_matrix_labels = [covariance_matrix_labels] * covmat.ndim
            self.parameter_covariance_matrix = CovarianceMatrix(
                covmat, covariance_matrix_labels
            )
            self.parameter_correlation_matrix = CorrelationMatrix(
                (covmat / errs).T / errs, covariance_matrix_labels
            )

            for pn in fitp.keys():
                uind = params.index(pn)  # Index of designmatrix
                un = 1.0 / (units[uind])  # Unit in designmatrix
                un *= u.s
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
                # NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            newparams = dict(zip(list(fitp.keys()), list(fitpv.values())))
            self.set_params(newparams)
            self.update_resids()
            # self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)

            # Compute the noise realizations if possible
            if not full_cov:
                noise_dims = self.model.noise_model_dimensions(self.toas)
                noise_resids = {}
                for comp in noise_dims:
                    # The first column of designmatrix is "offset", add 1 to match
                    # the indices of noise designmatrix
                    p0 = noise_dims[comp][0] + ntmpar + 1
                    p1 = p0 + noise_dims[comp][1]
                    noise_resids[comp] = np.dot(M[:, p0:p1], xhat[p0:p1]) * u.s
                    if debug:
                        setattr(self.resids, f"{comp}_M", (M[:, p0:p1], xhat[p0:p1]))
                        setattr(self.resids, f"{comp}_M_index", (p0, p1))
                self.resids.noise_resids = noise_resids
                if debug:
                    setattr(self.resids, "norm", norm)

        chi2 = self.resids.calc_chi2()
        self.update_model(chi2)

        return chi2


class WidebandTOAFitter(Fitter):  # Is GLSFitter the best here?
    """A class to for fitting TOAs and other independent measured data.

    Currently this fitter is only capable of fitting sets of TOAs in which every
    TOA has a DM measurement, and it should be constructed as
    ``WidebandTOAFitter(toas, model)``.

    Parameters
    ----------
    fit_data: data object or a tuple of data objects.
        The data to fit for. Generally this should be a single TOAs object containing
        DM information for every TOA.  If more than one data
        objects are provided, the size of ``fit_data`` has to match the
        ``fit_data_names``. In this fitter, the first fit data should be a TOAs object.
    model: a pint timing model instance
        The initial timing model for fitting.
    fit_data_names: list of str
        The names of the data fit for.
    additional_args: dict, optional
        The additional arguments for making residuals.
    """

    def __init__(
        self,
        fit_data,
        model,
        fit_data_names=["toa", "dm"],
        track_mode=None,
        additional_args={},
    ):
        self.model_init = model
        # Check input data and data_type
        self.fit_data_names = fit_data_names
        # convert the non tuple input to a tuple
        if not isinstance(fit_data, (tuple, list)):
            fit_data = [fit_data]
        if not isinstance(fit_data[0], TOAs):
            raise ValueError(
                f"The first data set should be a TOAs object but is {fit_data[0]}."
            )
        if len(fit_data_names) == 0:
            raise ValueError("Please specify the fit data.")
        if len(fit_data) > 1 and len(fit_data_names) != len(fit_data):
            raise ValueError(
                "If one more data sets are provided, the fit "
                "data have to match the fit data names."
            )
        self.fit_data = fit_data
        if track_mode is not None:
            if "toa" not in additional_args:
                additional_args["toa"] = {}
            additional_args["toa"]["track_mode"] = track_mode
        self.additional_args = additional_args
        # Get the makers for fitting parts.
        self.reset_model()
        self.resids_init = copy.deepcopy(self.resids)
        self.designmatrix_makers = [
            DesignMatrixMaker(data_resids.residual_type, data_resids.unit)
            for data_resids in self.resids.residual_objs.values()
        ]
        # Add noise design matrix maker
        self.noise_designmatrix_maker = DesignMatrixMaker("toa_noise", u.s)
        #
        self.covariancematrix_makers = [
            CovarianceMatrixMaker(data_resids.residual_type, data_resids.unit)
            for data_resids in self.resids.residual_objs.values()
        ]
        self.is_wideband = True
        self.method = "General_Data_Fitter"

    @property
    def toas(self):
        return self.fit_data[0]

    def make_combined_residuals(self, add_args={}, model=None):
        """Make the combined residuals between TOA residual and DM residual."""
        return WidebandTOAResiduals(
            self.toas,
            self.model if model is None else model,
            toa_resid_args=add_args.get("toa", {}),
            dm_resid_args=add_args.get("dm", {}),
        )

    def reset_model(self):
        """Reset the current model to the initial model."""
        self.model = copy.deepcopy(self.model_init)
        self.update_resids()
        self.fitresult = []

    def make_resids(self, model):
        """Update the residuals. Run after updating a model parameter."""
        return self.make_combined_residuals(add_args=self.additional_args, model=model)

    def get_designmatrix(self):
        design_matrixs = []
        fit_params = self.model.free_params
        if len(self.fit_data) == 1:
            design_matrixs.extend(
                dmatrix_maker(self.fit_data[0], self.model, fit_params, offset=True)
                for dmatrix_maker in self.designmatrix_makers
            )
        else:
            design_matrixs.extend(
                dmatrix_maker(self.fit_data[ii], self.model, fit_params, offset=True)
                for ii, dmatrix_maker in enumerate(self.designmatrix_makers)
            )
        return combine_design_matrices_by_quantity(design_matrixs)

    def get_noise_covariancematrix(self):
        # TODO This needs to be more general
        cov_matrixs = []
        if len(self.fit_data) == 1:
            cov_matrixs.extend(
                cmatrix_maker(self.fit_data[0], self.model)
                for cmatrix_maker in self.covariancematrix_makers
            )
        else:
            cov_matrixs.extend(
                cmatrix_maker(self.fit_data[ii], self.model)
                for ii, cmatrix_maker in enumerate(self.covariancematrix_makers)
            )
        return combine_covariance_matrix(cov_matrixs)

    def get_data_uncertainty(self, data_name, data_obj):
        """Get the data uncertainty from the data  object.

        Note
        ----
        TODO, make this more general.
        """
        func_map = {"toa": "get_errors", "dm": "get_dm_errors"}
        error_func_name = func_map[data_name]
        if hasattr(data_obj, error_func_name):
            return getattr(data_obj, error_func_name)()
        else:
            raise ValueError("No method to access data error is provided.")

    def scaled_all_sigma(self):
        """Scale all data's uncertainty.

        If the function of scaled_`data`_sigma is not given, it will just
        return the original data uncertainty.
        """
        scaled_sigmas = []
        sigma_units = []
        for ii, fd_name in enumerate(self.fit_data_names):
            func_name = f"scaled_{fd_name}_uncertainty"
            sigma_units.append(self.resids.residual_objs[fd_name].unit)
            if hasattr(self.model, func_name):
                scale_func = getattr(self.model, func_name)
                if len(self.fit_data) == 1:
                    scaled_sigmas.append(scale_func(self.fit_data[0]))
                else:
                    scaled_sigmas.append(scale_func(self.fit_data[ii]))
            else:
                if len(self.fit_data) == 1:
                    original_sigma = self.get_data_uncertainty(
                        fd_name, self.fit_data[0]
                    )
                else:
                    original_sigma = self.get_data_uncertainty(
                        fd_name, self.fit_data[ii]
                    )
                scaled_sigmas.append(original_sigma)

        scaled_sigmas_no_unit = []
        for ii, scaled_sigma in enumerate(scaled_sigmas):
            if hasattr(scaled_sigma, "unit"):
                scaled_sigmas_no_unit.append(scaled_sigma.to_value(sigma_units[ii]))
            else:
                scaled_sigmas_no_unit.append(scaled_sigma)
        return np.hstack(scaled_sigmas_no_unit)

    def fit_toas(self, maxiter=1, threshold=0, full_cov=False, debug=False):
        """Carry out a generalized least-squares fitting procedure.

        The algorithm here is essentially the same as used in
        :func:`pint.fitter.GLSFitter.fit_toas`. See that function
        for details.

        Parameters
        ----------
        maxiter: int
            How many times to run the linear least-squares fit, re-evaluating
            the derivatives at each step.
        threshold: float
            When to start discarding singular values. Default is 1-e14*max(M.shape).
        full_cov: bool
            full_cov determines which calculation is used.
        """
        # Maybe change the name to do_fit?
        # check that params of timing model have necessary components
        self.model.validate()
        self.model.validate_toas(self.toas)
        chi2 = 0
        for i in range(maxiter):
            fitp = self.model.get_params_dict("free", "quantity")
            fitpv = self.model.get_params_dict("free", "num")
            fitperrs = self.model.get_params_dict("free", "uncertainty")

            # Define the linear system
            d_matrix = self.get_designmatrix()
            M, params, units = (
                d_matrix.matrix,
                d_matrix.derivative_params,
                d_matrix.param_units,
            )

            # Get residuals and TOA uncertainties in seconds
            if i == 0:
                self.update_resids()
            # Since the residuals may not have the same unit. Thus the residual here
            # has no unit.
            residuals = self.resids._combined_resids

            # get any noise design matrices and weight vectors
            if not full_cov:
                # We assume the fit date type is toa
                Mn = self.noise_designmatrix_maker(self.toas, self.model)
                phi = self.model.noise_model_basis_weight(self.toas)
                phiinv = np.zeros(M.shape[1])
                if Mn is not None and phi is not None:
                    phiinv = np.concatenate((phiinv, 1 / phi))
                    new_d_matrix = combine_design_matrices_by_param(d_matrix, Mn)
                    M, params, units = (
                        new_d_matrix.matrix,
                        new_d_matrix.derivative_params,
                        new_d_matrix.param_units,
                    )

            ntmpar = len(fitp)

            # normalize the design matrix
            M, norm = normalize_designmatrix(M, params)
            self.fac = norm

            # compute covariance matrices
            if full_cov:
                cov = self.get_noise_covariancematrix().matrix
                cf = scipy.linalg.cho_factor(cov)
                cm = scipy.linalg.cho_solve(cf, M)
                mtcm = np.dot(M.T, cm)
                mtcy = np.dot(cm.T, residuals)

            else:
                phiinv /= norm**2
                Nvec = self.scaled_all_sigma() ** 2

                cinv = 1 / Nvec
                mtcm = np.dot(M.T, cinv[:, None] * M)
                mtcm += np.diag(phiinv)
                mtcy = np.dot(M.T, cinv * residuals)

            xhat, xvar = None, None
            if threshold <= 0:
                try:
                    c = scipy.linalg.cho_factor(mtcm)
                    xhat = scipy.linalg.cho_solve(c, mtcy)
                    xvar = scipy.linalg.cho_solve(c, np.eye(len(mtcy)))
                except scipy.linalg.LinAlgError:
                    xhat, xvar = None, None
            if xhat is None:
                U, s, Vt = scipy.linalg.svd(mtcm, full_matrices=False)

                bad = np.where(s <= threshold * s[0])[0]
                s[bad] = np.inf
                for c in bad:
                    bad_col = Vt[c, :]
                    bad_col /= abs(bad_col).max()
                    bad_combination = " ".join(
                        [
                            f"{co}*{p}"
                            for (co, p) in reversed(sorted(zip(bad_col, params)))
                            if abs(co) > threshold
                        ]
                    )
                    warn(
                        f"Parameter degeneracy; the following combination of parameters yields "
                        f"almost no change: {bad_combination}",
                        DegeneracyWarning,
                    )

                xvar = np.dot(Vt.T / s, Vt)
                xhat = np.dot(Vt.T, np.dot(U.T, mtcy) / s)
            newres = residuals - np.dot(M, xhat)
            # compute linearized chisq
            if full_cov:
                chi2 = np.dot(newres, scipy.linalg.cho_solve(cf, newres))
            else:
                chi2 = np.dot(newres, cinv * newres) + np.dot(xhat, phiinv * xhat)

            # compute absolute estimates, normalized errors, covariance matrix
            dpars = xhat / norm
            errs = np.sqrt(np.diag(xvar)) / norm
            covmat = (xvar / norm).T / norm
            # TODO: seems like doing this on every iteration is wasteful, and we should just do it once and then update the matrix
            covariance_matrix_labels = {
                param: (i, i + 1, unit)
                for i, (param, unit) in enumerate(zip(params, units))
            }
            # covariance matrix is 2D and symmetric
            covariance_matrix_labels = [covariance_matrix_labels] * covmat.ndim
            self.parameter_covariance_matrix = CovarianceMatrix(
                covmat, covariance_matrix_labels
            )
            self.parameter_correlation_matrix = CorrelationMatrix(
                (covmat / errs).T / errs, covariance_matrix_labels
            )

            # self.covariance_matrix = covmat
            # self.correlation_matrix = (covmat / errs).T / errs

            for pn in fitp.keys():
                uind = params.index(pn)  # Index of designmatrix
                # Here we use design matrix's label, so the unit goes to normal.
                # instead of un = 1 / (units[uind])
                un = units[uind]
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
                # NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            newparams = dict(zip(list(fitp.keys()), list(fitpv.values())))
            self.set_params(newparams)
            self.update_resids()
            # self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)

            # Compute the noise realizations if possible
            if not full_cov:
                noise_dims = self.model.noise_model_dimensions(self.toas)
                noise_resids = {}
                for comp in noise_dims:
                    # The first column of designmatrix is "offset", add 1 to match
                    # the indices of noise designmatrix
                    p0 = noise_dims[comp][0] + ntmpar + 1
                    p1 = p0 + noise_dims[comp][1]
                    noise_resids[comp] = np.dot(M[:, p0:p1], xhat[p0:p1]) * u.s
                    if debug:
                        setattr(self.resids, f"{comp}_M", (M[:, p0:p1], xhat[p0:p1]))
                        setattr(self.resids, f"{comp}_M_index", (p0, p1))
                self.resids.noise_resids = noise_resids
                if debug:
                    setattr(self.resids, "norm", norm)

        self.update_model(chi2)

        return chi2


class LMFitter(Fitter):
    def fit_toas(
        self,
        maxiter=50,
        *,
        min_chi2_decrease=1e-3,
        lambda_factor_decrease=2,
        lambda_factor_increase=3,
        lambda_factor_invalid=10,
        threshold=1e-14,
        min_lambda=0.5,
        debug=False,
    ):
        current_state = self.create_state()
        try:
            try:
                current_state.chi2
            except ValueError as e:
                raise ValueError("Initial configuration is invalid") from e
            self.converged = False
            lambda_ = min_lambda
            for i in range(maxiter):
                lf = lambda_ if lambda_ > min_lambda else 0
                # Attempt: do not scale the phiinv penalty factor by lambda
                A = current_state.mtcm + lf * np.diag(np.diag(current_state.mtcmplain))
                b = current_state.mtcy
                ill_conditioned = False
                if threshold is None:
                    dx = scipy.linalg.solve(A, b, assume_a="pos")
                else:
                    U, s, Vt = scipy.linalg.svd(A, full_matrices=False)
                    log.trace(
                        f"Iteration {i}: Condition number for lambda_ = {lambda_} is {s[0]/s[-1]}"
                    )

                    bad = np.where(s <= threshold * s[0])[0]
                    s[bad] = np.inf
                    for c in bad:
                        ill_conditioned = True
                        # FIXME: maybe don't stop while ill-conditioned? Always try increasing lambda?
                        bad_col = Vt[c, :]
                        bad_col /= abs(bad_col).max()
                        bad_combination = " ".join(
                            [
                                f"{co}*{p}"
                                for (co, p) in reversed(
                                    sorted(zip(bad_col, current_state.params))
                                )
                                if abs(co) > threshold
                            ]
                        )
                        warn(
                            f"Parameter degeneracy; the following combination of parameters yields "
                            f"almost no change: {bad_combination}",
                            DegeneracyWarning,
                        )

                    dx = np.dot(Vt.T, np.dot(U.T, b) / s)

                step = dx / current_state.norm

                # FIXME: catch problems due to non-invertibility?
                # FIXME: predicted (linear) chi-squared decrease can check how well the
                # derivative matches the function and guide changes in lambda_
                # predicted_chi2 = current_state.predicted_chi2(dx)
                log.trace(f"Iteration {i}: Trying step with lambda_ = {lambda_}")
                new_state = current_state.take_step(step)
                try:
                    chi2_decrease = current_state.chi2 - new_state.chi2
                    if chi2_decrease < -min_chi2_decrease:
                        lambda_ *= (
                            lambda_factor_invalid
                            if ill_conditioned
                            else lambda_factor_increase
                        )
                        log.trace(
                            f"Iteration {i}: chi2 increased from {current_state.chi2} "
                            f"to {new_state.chi2} increasing lambda to {lambda_}"
                        )
                    elif chi2_decrease < 0:
                        log.info(
                            f"Iteration {i}: chi2 increased but only by {-chi2_decrease}, stopping."
                        )
                        self.converged = True
                        break
                    elif chi2_decrease < min_chi2_decrease:
                        log.debug(
                            f"Iteration {i}: chi2 decreased only by {chi2_decrease}, updating "
                            f"state and stopping."
                        )
                        current_state = new_state
                        self.converged = True
                        break
                    else:
                        lambda_ = max(lambda_ / lambda_factor_decrease, min_lambda)
                        log.debug(
                            f"Iteration {i}: Updating state, chi2 goes down by {chi2_decrease} "
                            f"from {current_state.chi2} "
                            f"to {new_state.chi2}; decreasing lambda to "
                            f"{lambda_ if lambda_ > min_lambda else 0}"
                        )
                        current_state = new_state
                except InvalidModelParameters as e:
                    lambda_ *= lambda_factor_invalid
                    log.debug(
                        f"Iteration {i}: Step too aggressive, increasing lambda_ "
                        f"to {lambda_}: {e}"
                    )
            else:
                log.warning(
                    f"Maximum number of iterations ({maxiter}) reached, stopping "
                    f"without convergence."
                )
            self.iterations = i
        except KeyboardInterrupt:
            # could be a finally I suppose? but I'm not sure we want to update if something
            # seriou went wrong.
            log.info("KeyboardInterrupt detected, updating Fitter")
            self.update_from_state(current_state, debug=debug)
            raise
        self.update_from_state(current_state, debug=debug)
        return self.converged


class WidebandLMFitter(LMFitter):
    """Fitter for wideband data based on Levenberg-Marquardt.

    This should carry out a more reliable fitting process than the plain
    WidebandTOAFitter, and a more efficient one than WidebandDownhillFitter.
    Unfortunately it doesn't.
    """

    def __init__(self, toas, model, track_mode=None, residuals=None, add_args=None):
        self.method = "downhill_wideband"
        self.full_cov = False
        self.threshold = 0
        self.add_args = {} if add_args is None else add_args
        super().__init__(
            toas=toas, model=model, residuals=residuals, track_mode=track_mode
        )
        self.is_wideband = True

    def make_resids(self, model):
        return WidebandTOAResiduals(
            self.toas,
            model,
            toa_resid_args=self.add_args.get("toa", {}),
            dm_resid_args=self.add_args.get("dm", {}),
        )

    def create_state(self):
        return WidebandState(
            self, self.model, full_cov=self.full_cov, threshold=self.threshold
        )

    def fit_toas(self, maxiter=50, full_cov=False, debug=False, **kwargs):
        self.full_cov = full_cov
        # FIXME: set up noise residuals et cetera
        return super().fit_toas(maxiter=maxiter, debug=debug, **kwargs)

    def update_from_state(self, state, debug=False):
        # Nicer not to keep this if we have a choice, it introduces reference cycles
        self.current_state = state
        self.model = state.model
        self.resids = state.resids
        self.parameter_covariance_matrix = state.parameter_covariance_matrix
        self.errors = np.sqrt(np.diag(self.parameter_covariance_matrix.matrix))
        for p, e in zip(state.params, self.errors):
            try:
                log.trace(f"Setting {getattr(self.model, p)} uncertainty to {e}")
                pm = getattr(self.model, p)
            except AttributeError:
                if p != "Offset":
                    log.warning(f"Unexpected parameter {p}")
            else:
                pm.uncertainty = e * pm.units
        # self.parameter_correlation_matrix = (
        #    self.parameter_covariance_matrix / self.errors
        # ).T / self.errors
        self.parameter_correlation_matrix = CorrelationMatrix(
            (self.parameter_covariance_matrix.matrix / self.errors).T / self.errors,
            self.parameter_covariance_matrix.axis_labels,
        )

        self.update_model(state.chi2)
        # Compute the noise realizations if possible
        if not self.full_cov:
            noise_dims = self.model.noise_model_dimensions(self.toas)
            noise_resids = {}
            ntmpar = len(self.model.free_params)
            for comp in noise_dims:
                # The first column of designmatrix is "offset", add 1 to match
                # the indices of noise designmatrix
                p0 = noise_dims[comp][0] + ntmpar + 1
                p1 = p0 + noise_dims[comp][1]
                noise_resids[comp] = np.dot(state.M[:, p0:p1], state.xhat[p0:p1]) * u.s
                if debug:
                    setattr(
                        self.resids, f"{comp}_M", (state.M[:, p0:p1], state.xhat[p0:p1])
                    )
                    setattr(self.resids, f"{comp}_M_index", (p0, p1))
            self.resids.noise_resids = noise_resids
            if debug:
                setattr(self.resids, "norm", state.norm)
