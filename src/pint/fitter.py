from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy

import astropy.constants as const
import astropy.units as u
import numpy as np
import pint.utils
import scipy.linalg as sl
import scipy.optimize as opt
from astropy import log
from pint.toa import TOAs
from pint import Tsun
from pint.utils import FTest
from pint.pint_matrix import (
    DesignMatrixMaker,
    CovarianceMatrixMaker,
    combine_design_matrices_by_quantity,
    combine_design_matrices_by_param,
    combine_covariance_matrix,
)

import pint.residuals as pr

from pint.models.parameter import (
    AngleParameter,
    boolParameter,
    floatParameter,
    prefixParameter,
    strParameter,
    MJDParameter,
)
from pint.models.pulsar_binary import PulsarBinary
from pint.residuals import Residuals
from pint.utils import FTest

__all__ = ["Fitter", "PowellFitter", "GLSFitter", "WLSFitter"]


class Fitter(object):
    """ Base class for fitter.

    The fitting function should be defined as the fit_toas() method.

    Note that the Fitter object makes a deepcopy of the model, so changes to the model
    will not be noticed after the Fitter has been instantiated!  Use Fitter.model instead.

    The Fitter also caches a copy of the original model so it can be restored with reset_model()



    Parameters
    ----------
    toas : a pint TOAs instance
        The input toas.
    model : a pint timing model instance
        The initial timing model for fitting.
    """

    def __init__(self, toas, model, residuals=None):
        self.toas = toas
        self.model_init = model
        if residuals is None:
            self.resids_init = pr.Residuals(toas=toas, model=model)
            self.reset_model()
        else:
            # residuals were provided, we're just going to use them
            # probably using GLSFitter to compute a chi-squared
            self.model = copy.deepcopy(self.model_init)
            self.resids = residuals
            self.fitresult = []
        self.method = None

    def reset_model(self):
        """Reset the current model to the initial model."""
        self.model = copy.deepcopy(self.model_init)
        self.update_resids()
        self.fitresult = []

    def update_resids(self):
        """Update the residuals. Run after updating a model parameter."""
        self.resids = pr.Residuals(toas=self.toas, model=self.model)

    def set_fitparams(self, *params):
        """Update the "frozen" attribute of model parameters.

        Ex. fitter.set_fitparams('F0','F1')
        """
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

        for p in self.model.params:
            getattr(self.model, p).frozen = p not in fit_params_name

    def get_allparams(self):
        """Return a dict of all param names and values."""
        return collections.OrderedDict(
            (k, getattr(self.model, k).quantity) for k in self.model.params_ordered
        )

    def get_fitparams(self):
        """Return a dict of fittable param names and quantity."""
        return collections.OrderedDict(
            (k, getattr(self.model, k))
            for k in self.model.params
            if not getattr(self.model, k).frozen
        )

    def get_fitparams_num(self):
        """Return a dict of fittable param names and numeric values."""
        return collections.OrderedDict(
            (k, getattr(self.model, k).value)
            for k in self.model.params
            if not getattr(self.model, k).frozen
        )

    def get_fitparams_uncertainty(self):
        return collections.OrderedDict(
            (k, getattr(self.model, k).uncertainty_value)
            for k in self.model.params
            if not getattr(self.model, k).frozen
        )

    def set_params(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        Ex. fitter.set_params({'F0':60.1,'F1':-1.3e-15})
        """
        # In Powell fitter this sometimes fails because after some iterations the values change from
        # plain float to Quantities. No idea why.
        if len(fitp.values()) < 1:
            return
        if isinstance(list(fitp.values())[0], u.Quantity):
            for k, v in fitp.items():
                getattr(self.model, k).value = v.value
        else:
            for k, v in fitp.items():
                getattr(self.model, k).value = v

    def set_param_uncertainties(self, fitp):
        for k, v in fitp.items():
            parunit = getattr(self.model, k).units
            getattr(self.model, k).uncertainty = v * parunit

    def get_designmatrix(self):
        return self.model.designmatrix(toas=self.toas, incfrozen=False, incoffset=True)

    def minimize_func(self, x, *args):
        """Wrapper function for the residual class, meant to be passed to
        scipy.optimize.minimize. The function must take a single list of input
        values, x, and a second optional tuple of input arguments.  It returns
        a quantity to be minimized (in this case chi^2).
        """
        self.set_params({k: v for k, v in zip(args, x)})
        self.update_resids()
        # Return chi^2
        return self.resids.chi2

    def fit_toas(self, maxiter=None):
        raise NotImplementedError

    def plot(self):
        """Make residuals plot"""
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
        except:
            psr = self.model.PSRJ
        else:
            psr = "Residuals"
        ax.set_title(psr)
        ax.grid(True)
        plt.show()

    def get_summary(self, nodmx=False):
        """Return a human-readable summary of the Fitter results.

        Parameters
        ----------
        nodmx : bool
            Set to True to suppress printing DMX parameters in summary
        """

        # Need to check that fit has been done first!
        if not hasattr(self, "covariance_matrix"):
            log.warning(
                "fit_toas() has not been run, so pre-fit and post-fit will be the same!"
            )

        from uncertainties import ufloat
        import uncertainties.umath as um

        # First, print fit quality metrics
        s = "Fitted model using {} method with {} free parameters to {} TOAs\n".format(
            self.method, len(self.get_fitparams()), self.toas.ntoas
        )
        s += "Prefit residuals Wrms = {}, Postfit residuals Wrms = {}\n".format(
            self.resids_init.rms_weighted(), self.resids.rms_weighted()
        )
        s += "Chisq = {:.3f} for {} d.o.f. for reduced Chisq of {:.3f}\n".format(
            self.resids.chi2, self.resids.dof, self.resids.chi2_reduced
        )
        s += "\n"

        # to handle all parameter names, determine the longest length for the first column
        longestName = 0  # optionally specify the minimum length here instead of 0
        for pn in list(self.get_allparams().keys()):
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
        for pn in list(self.get_allparams().keys()):
            if nodmx and pn.startswith("DMX"):
                continue
            prefitpar = getattr(self.model_init, pn)
            par = getattr(self.model, pn)
            if par.value is not None:
                if isinstance(par, strParameter):
                    s += ("{:" + spacingName + "s} {:>20s} {:28s} {}\n").format(
                        pn, prefitpar.value, "", par.units
                    )
                elif isinstance(par, AngleParameter):
                    # Add special handling here to put uncertainty into arcsec
                    if par.frozen:
                        s += ("{:" + spacingName + "s} {:>20s} {:>28s} {} \n").format(
                            pn, str(prefitpar.quantity), "", par.units
                        )
                    else:
                        if par.units == u.hourangle:
                            uncertainty_unit = pint.hourangle_second
                        else:
                            uncertainty_unit = u.arcsec
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
                        pn, prefitpar.print_quantity(prefitpar.value), "", par.units
                    )
                else:
                    # Assume a numerical parameter
                    if par.frozen:
                        if par.name == "START":
                            if prefitpar.value is None:
                                s += (
                                    "{:" + spacingName + "s} {:20s} {:28g} {} \n"
                                ).format(pn, " ", par.value, par.units)
                            else:
                                s += (
                                    "{:" + spacingName + "s} {:20g} {:28g} {} \n"
                                ).format(pn, prefitpar.value, par.value, par.units)
                        elif par.name == "FINISH":
                            if prefitpar.value is None:
                                s += (
                                    "{:" + spacingName + "s} {:20s} {:28g} {} \n"
                                ).format(pn, " ", par.value, par.units)
                            else:
                                s += (
                                    "{:" + spacingName + "s} {:20g} {:28g} {} \n"
                                ).format(pn, prefitpar.value, par.value, par.units)
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
        # Now print some useful derived parameters
        s += "\nDerived Parameters:\n"
        if hasattr(self.model, "F0"):
            F0 = self.model.F0.quantity
            if not self.model.F0.frozen:
                p, perr = pint.utils.pferrs(F0, self.model.F0.uncertainty)
                s += "Period = {} +/- {}\n".format(p.to(u.s), perr.to(u.s))
            else:
                s += "Period = {}\n".format((1.0 / F0).to(u.s))
        if hasattr(self.model, "F1"):
            F1 = self.model.F1.quantity
            if not any([self.model.F1.frozen, self.model.F0.frozen]):
                p, perr, pd, pderr = pint.utils.pferrs(
                    F0, self.model.F0.uncertainty, F1, self.model.F1.uncertainty
                )
                s += "Pdot = {} +/- {}\n".format(
                    pd.to(u.dimensionless_unscaled), pderr.to(u.dimensionless_unscaled)
                )
                brakingindex = 3
                s += "Characteristic age = {:.4g} (braking index = {})\n".format(
                    pint.utils.pulsar_age(F0, F1, n=brakingindex), brakingindex
                )
                s += "Surface magnetic field = {:.3g}\n".format(
                    pint.utils.pulsar_B(F0, F1)
                )
                s += "Magnetic field at light cylinder = {:.4g}\n".format(
                    pint.utils.pulsar_B_lightcyl(F0, F1)
                )
                I_NS = I = 1.0e45 * u.g * u.cm ** 2
                s += "Spindown Edot = {:.4g} (I={})\n".format(
                    pint.utils.pulsar_edot(F0, F1, I=I_NS), I_NS
                )

        if hasattr(self.model, "PX"):
            if not self.model.PX.frozen:
                s += "\n"
                px = ufloat(
                    self.model.PX.quantity.to(u.arcsec).value,
                    self.model.PX.uncertainty.to(u.arcsec).value,
                )
                s += "Parallax distance = {:.3uP} pc\n".format(1.0 / px)

        # Now binary system derived parameters
        binary = None
        for x in self.model.components:
            if x.startswith("Binary"):
                binary = x
        if binary is not None:
            s += "\n"
            s += "Binary model {}\n".format(binary)

            if binary.startswith("BinaryELL1"):
                if not any(
                    [
                        self.model.EPS1.frozen,
                        self.model.EPS2.frozen,
                        self.model.TASC.frozen,
                        self.model.PB.frozen,
                    ]
                ):
                    eps1 = ufloat(
                        self.model.EPS1.quantity.value,
                        self.model.EPS1.uncertainty.value,
                    )
                    eps2 = ufloat(
                        self.model.EPS2.quantity.value,
                        self.model.EPS2.uncertainty.value,
                    )
                    tasc = ufloat(
                        # This is a time in MJD
                        self.model.TASC.quantity.mjd,
                        self.model.TASC.uncertainty.to(u.d).value,
                    )
                    pb = ufloat(
                        self.model.PB.quantity.to(u.d).value,
                        self.model.PB.uncertainty.to(u.d).value,
                    )
                    s += "Conversion from ELL1 parameters:\n"
                    ecc = um.sqrt(eps1 ** 2 + eps2 ** 2)
                    s += "ECC = {:P}\n".format(ecc)
                    om = um.atan2(eps1, eps2) * 180.0 / np.pi
                    if om < 0.0:
                        om += 360.0
                    s += "OM  = {:P}\n".format(om)
                    t0 = tasc + pb * om / 360.0
                    s += "T0  = {:SP}\n".format(t0)

                    s += pint.utils.ELL1_check(
                        self.model.A1.quantity,
                        ecc.nominal_value,
                        self.resids.rms_weighted(),
                        self.toas.ntoas,
                        outstring=True,
                    )
                    s += "\n"

                # Masses and inclination
                if not any([self.model.PB.frozen, self.model.A1.frozen]):
                    pbs = ufloat(
                        self.model.PB.quantity.to(u.s).value,
                        self.model.PB.uncertainty.to(u.s).value,
                    )
                    a1 = ufloat(
                        self.model.A1.quantity.to(pint.ls).value,
                        self.model.A1.uncertainty.to(pint.ls).value,
                    )
                    fm = 4.0 * np.pi ** 2 * a1 ** 3 / (4.925490947e-6 * pbs ** 2)
                    s += "Mass function = {:SP} Msun\n".format(fm)
                    mcmed = pint.utils.companion_mass(
                        self.model.PB.quantity,
                        self.model.A1.quantity,
                        inc=60.0 * u.deg,
                        mpsr=1.4 * u.solMass,
                    )
                    mcmin = pint.utils.companion_mass(
                        self.model.PB.quantity,
                        self.model.A1.quantity,
                        inc=90.0 * u.deg,
                        mpsr=1.4 * u.solMass,
                    )
                    s += "Companion mass min, median (assuming Mpsr = 1.4 Msun) = {:.4f}, {:.4f} Msun\n".format(
                        mcmin, mcmed
                    )

                if hasattr(self.model, "SINI"):
                    try:
                        # Put this in a try in case SINI is UNSET or an illegal value
                        if not self.model.SINI.frozen:
                            si = ufloat(
                                self.model.SINI.quantity.value,
                                self.model.SINI.uncertainty.value,
                            )
                            s += "From SINI in model:\n"
                            s += "    cos(i) = {:SP}\n".format(um.sqrt(1 - si ** 2))
                            s += "    i = {:SP} deg\n".format(
                                um.asin(si) * 180.0 / np.pi
                            )

                        psrmass = pint.utils.pulsar_mass(
                            self.model.PB.quantity,
                            self.model.A1.quantity,
                            self.model.M2.quantity,
                            np.arcsin(self.model.SINI.quantity),
                        )
                        s += "Pulsar mass (Shapiro Delay) = {}".format(psrmass)
                    except:
                        pass

        return s

    def print_summary(self):
        """Write a summary of the TOAs to stdout."""
        print(self.get_summary())

    def get_covariance_matrix(self, with_phase=False, pretty_print=False, prec=3):
        """Show the parameter covariance matrix post-fit.
        If with_phase, then show and return the phase column as well.
        If pretty_print, then also pretty-print on stdout the matrix.
        prec is the precision of the floating point results.
        """
        if hasattr(self, "covariance_matrix"):
            fps = list(self.get_fitparams().keys())
            cm = self.covariance_matrix
            if with_phase:
                fps = ["PHASE"] + fps
            else:
                cm = cm[1:, 1:]
            if pretty_print:
                lens = [max(len(fp) + 2, prec + 8) for fp in fps]
                maxlen = max(lens)
                print("\nParameter covariance matrix:")
                line = "{0:^{width}}".format("", width=maxlen)
                for fp, ln in zip(fps, lens):
                    line += "{0:^{width}}".format(fp, width=ln)
                print(line)
                for ii, fp1 in enumerate(fps):
                    line = "{0:^{width}}".format(fp1, width=maxlen)
                    for jj, (fp2, ln) in enumerate(zip(fps[: ii + 1], lens[: ii + 1])):
                        line += "{0: {width}.{prec}e}".format(
                            cm[ii, jj], width=ln, prec=prec
                        )
                    print(line)
                print("\n")
            return cm
        else:
            log.error("You must run .fit_toas() before accessing the covariance matrix")
            raise AttributeError

    def get_correlation_matrix(self, with_phase=False, pretty_print=False, prec=3):
        """Show the parameter correlation matrix post-fit.
        If with_phase, then show and return the phase column as well.
        If pretty_print, then also pretty-print on stdout the matrix.
        prec is the precision of the floating point results.
        """
        if hasattr(self, "correlation_matrix"):
            fps = list(self.get_fitparams().keys())
            cm = self.correlation_matrix
            if with_phase:
                fps = ["PHASE"] + fps
            else:
                cm = cm[1:, 1:]
            if pretty_print:
                lens = [max(len(fp) + 2, prec + 4) for fp in fps]
                maxlen = max(lens)
                print("\nParameter correlation matrix:")
                line = "{0:^{width}}".format("", width=maxlen)
                for fp, ln in zip(fps, lens):
                    line += "{0:^{width}}".format(fp, width=ln)
                print(line)
                for ii, fp1 in enumerate(fps):
                    line = "{0:^{width}}".format(fp1, width=maxlen)
                    for jj, (fp2, ln) in enumerate(zip(fps, lens)):
                        line += "{0:^{width}.{prec}f}".format(
                            cm[ii, jj], width=ln, prec=prec
                        )
                    print(line)
                print("\n")
            return cm
        else:
            log.error(
                "You must run .fit_toas() before accessing the correlation matrix"
            )
            raise AttributeError

    def ftest(self, parameter, component, remove=False, full_output=False):
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

        Returns
        --------
        dictionary

            ft : Float
                F-test significance value for the model with the larger number of
                components over the other. Computed with pint.utils.FTest().

            resid_rms_test : Float (Quantity)
                If full_output is True, returns the RMS of the residuals of the tested model
                fit. Will be in units of microseconds as an astropy quantity.

            resid_wrms_test : Float (Quantity)
                If full_output is True, returns the Weighted RMS of the residuals of the tested model
                fit. Will be in units of microseconds as an astropy quantity.

            chi2_test : Float
                If full_output is True, returns the chi-squared of the tested model.

            dof_test : Int
                If full_output is True, returns the degrees of freedom of the tested model.
        """
        # Copy the fitter that we do not change the initial model and fitter
        fitter_copy = copy.deepcopy(self)
        # Number of times to run the fit
        NITS = 1
        # We need the original degrees of freedome and chi-squared value
        # Because this applies to nested models, model 1 must always have fewer parameters
        if remove:
            dof_2 = self.resids.get_dof()
            chi2_2 = self.resids.calc_chi2()
        else:
            dof_1 = self.resids.get_dof()
            chi2_1 = self.resids.calc_chi2()
        # Single inputs are converted to lists to handle arb. number of parameteres
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
            fitter_copy.fit_toas(NITS)
            # Now get the new values
            dof_1 = fitter_copy.resids.get_dof()
            chi2_1 = fitter_copy.resids.calc_chi2()
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
                    if parameter[ii].name in check_params.keys():
                        if parameter[ii].value == 0.0:
                            log.warning(
                                "Default value for %s cannot be 0, resetting to %s"
                                % (parameter[ii].name, check_params[parameter[ii].name])
                            )
                            parameter[ii].value = check_params[parameter[ii].name]
                    getattr(
                        fitter_copy.model, "{:}".format(parameter[ii].name)
                    ).value = parameter[ii].value
                # If not, add it to the model
                else:
                    fitter_copy.model.components[component[ii]].add_param(
                        parameter[ii], setup=True
                    )
            # validate and setup model
            fitter_copy.model.validate()
            fitter_copy.model.setup()
            # Now refit
            fitter_copy.fit_toas(NITS)
            # Now get the new values
            dof_2 = fitter_copy.resids.get_dof()
            chi2_2 = fitter_copy.resids.calc_chi2()
        # Now run the actual F-test
        ft = FTest(chi2_1, dof_1, chi2_2, dof_2)

        if full_output:
            if remove:
                dof_test = dof_1
                chi2_test = chi2_1
            else:
                dof_test = dof_2
                chi2_test = chi2_2
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
            return {"ft": ft}


class PowellFitter(Fitter):
    """A class for Scipy Powell fitting method. This method searches over
       parameter space. It is a relative basic method.
    """

    def __init__(self, toas, model):
        super(PowellFitter, self).__init__(toas, model)
        self.method = "Powell"

    def fit_toas(self, maxiter=20):
        # check that params of timing model have necessary components
        self.model.maskPar_has_toas_check(self.toas)
        # Initial guesses are model params
        fitp = self.get_fitparams_num()
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

        # Update START/FINISH params
        self.model.START.value = self.toas.first_MJD
        self.model.FINISH.value = self.toas.last_MJD

        return self.resids.chi2


class WLSFitter(Fitter):
    """
       A class for weighted least square fitting method. The design matrix is
       required.
    """

    def __init__(self, toas, model):
        super(WLSFitter, self).__init__(toas=toas, model=model)
        self.method = "weighted_least_square"

    def fit_toas(self, maxiter=1, threshold=False):
        """Run a linear weighted least-squared fitting method"""
        # check that params of timing model have necessary components
        self.model.maskPar_has_toas_check(self.toas)
        chi2 = 0
        for i in range(maxiter):
            fitp = self.get_fitparams()
            fitpv = self.get_fitparams_num()
            fitperrs = self.get_fitparams_uncertainty()
            # Define the linear system
            M, params, units, scale_by_F0 = self.get_designmatrix()
            # Get residuals and TOA uncertainties in seconds
            self.update_resids()
            residuals = self.resids.time_resids.to(u.s).value
            Nvec = self.toas.get_errors().to(u.s).value

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
            fac = M.std(axis=0)
            fac[0] = 1.0
            M /= fac
            # Singular value decomp of design matrix:
            #   M = U s V^T
            # Dimensions:
            #   M, U are Ntoa x Nparam
            #   s is Nparam x Nparam diagonal matrix encoded as 1-D vector
            #   V^T is Nparam x Nparam
            U, s, Vt = sl.svd(M, full_matrices=False)
            # Note, here we could do various checks like report
            # matrix condition number or zero out low singular values.
            # print 'log_10 cond=', np.log10(s.max()/s.min())
            # Note, Check the threshold from data precision level.Borrowed from
            # np Curve fit.
            if threshold:
                threshold_val = np.finfo(np.longdouble).eps * max(M.shape) * s[0]
                s[s < threshold_val] = 0.0
            # Sigma = np.dot(Vt.T / s, U.T)
            # The post-fit parameter covariance matrix
            #   Sigma = V s^-2 V^T
            Sigma = np.dot(Vt.T / (s ** 2), Vt)
            # Parameter uncertainties. Scale by fac recovers original units.
            errs = np.sqrt(np.diag(Sigma)) / fac
            # covariance matrix stuff (for randomized models in pintk)
            sigma_var = (Sigma / fac).T / fac
            errors = np.sqrt(np.diag(sigma_var))
            sigma_cov = (sigma_var / errors).T / errors
            # covariance matrix = variances in diagonal, used for gaussian random models
            self.covariance_matrix = sigma_var
            # correlation matrix = 1s in diagonal, use for comparison to tempo/tempo2 cov matrix
            self.correlation_matrix = sigma_cov
            self.fac = fac
            self.errors = errors

            # The delta-parameter values
            #   dpars = V s^-1 U^T r
            # Scaling by fac recovers original units
            dpars = np.dot(Vt.T, np.dot(U.T, residuals) / s) / fac
            for ii, pn in enumerate(fitp.keys()):
                uind = params.index(pn)  # Index of designmatrix
                un = 1.0 / (units[uind])  # Unit in designmatrix
                if scale_by_F0:
                    un *= u.s
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
                # NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            chi2 = self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)

        # Update START/FINISH params
        self.model.START.value = self.toas.first_MJD
        self.model.FINISH.value = self.toas.last_MJD

        return chi2


class GLSFitter(Fitter):
    """
       A class for weighted least square fitting method. The design matrix is
       required.
    """

    def __init__(self, toas=None, model=None, residuals=None):
        super(GLSFitter, self).__init__(toas=toas, model=model, residuals=residuals)
        self.method = "generalized_least_square"

    def fit_toas(self, maxiter=1, threshold=False, full_cov=False):
        """Run a Generalized least-squared fitting method

        If maxiter is less than one, no fitting is done, just the
        chi-squared computation. In this case, you must provide the residuals
        argument.

        If maxiter is one or more, so fitting is actually done, the
        chi-squared value returned is only approximately the chi-squared
        of the improved(?) model. In fact it is the chi-squared of the
        solution to the linear fitting problem, and the full non-linear
        model should be evaluated and new residuals produced if an accurate
        chi-squared is desired.

        A first attempt is made to solve the fitting problem by Cholesky
        decomposition, but if this fails singular value decomposition is
        used instead. In this case singular values below threshold are removed.

        full_cov determines which calculation is used. If true, the full
        covariance matrix is constructed and the calculation is relatively
        straightforward but the full covariance matrix may be enormous.
        If false, an algorithm is used that takes advantage of the structure
        of the covariance matrix, based on information provided by the noise
        model. The two algorithms should give the same result to numerical
        accuracy where they both can be applied.
        """
        # check that params of timing model have necessary components
        self.model.maskPar_has_toas_check(self.toas)
        chi2 = 0
        for i in range(max(maxiter, 1)):
            fitp = self.get_fitparams()
            fitpv = self.get_fitparams_num()
            fitperrs = self.get_fitparams_uncertainty()

            # Define the linear system
            M, params, units, scale_by_F0 = self.get_designmatrix()

            # Get residuals and TOA uncertainties in seconds
            if i == 0:
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

            # normalize the design matrix
            norm = np.sqrt(np.sum(M ** 2, axis=0))
            ntmpar = len(fitp)
            if M.shape[1] > ntmpar:
                norm[ntmpar:] = 1
            if np.any(norm == 0):
                # Make this a LinAlgError so it looks like other bad matrixness
                raise sl.LinAlgError(
                    "One or more of the design-matrix columns is null."
                )
            M /= norm

            # compute covariance matrices
            if full_cov:
                cov = self.model.toa_covariance_matrix(self.toas)
                cf = sl.cho_factor(cov)
                cm = sl.cho_solve(cf, M)
                mtcm = np.dot(M.T, cm)
                mtcy = np.dot(cm.T, residuals)

            else:
                Nvec = self.model.scaled_toa_uncertainty(self.toas).to(u.s).value ** 2
                cinv = 1 / Nvec
                mtcm = np.dot(M.T, cinv[:, None] * M)
                mtcm += np.diag(phiinv)
                mtcy = np.dot(M.T, cinv * residuals)

            if maxiter > 0:
                try:
                    c = sl.cho_factor(mtcm)
                    xhat = sl.cho_solve(c, mtcy)
                    xvar = sl.cho_solve(c, np.eye(len(mtcy)))
                except sl.LinAlgError:
                    U, s, Vt = sl.svd(mtcm, full_matrices=False)

                    if threshold:
                        threshold_val = (
                            np.finfo(np.longdouble).eps * max(M.shape) * s[0]
                        )
                        s[s < threshold_val] = 0.0

                    xvar = np.dot(Vt.T / s, Vt)
                    xhat = np.dot(Vt.T, np.dot(U.T, mtcy) / s)
                newres = residuals - np.dot(M, xhat)
                # compute linearized chisq
                if full_cov:
                    chi2 = np.dot(newres, sl.cho_solve(cf, newres))
                else:
                    chi2 = np.dot(newres, cinv * newres) + np.dot(xhat, phiinv * xhat)
            else:
                newres = residuals
                if full_cov:
                    chi2 = np.dot(newres, sl.cho_solve(cf, newres))
                else:
                    chi2 = np.dot(newres, cinv * newres)
                return chi2

            # compute absolute estimates, normalized errors, covariance matrix
            dpars = xhat / norm
            errs = np.sqrt(np.diag(xvar)) / norm
            covmat = (xvar / norm).T / norm
            self.covariance_matrix = covmat
            self.correlation_matrix = (covmat / errs).T / errs

            for ii, pn in enumerate(fitp.keys()):
                uind = params.index(pn)  # Index of designmatrix
                un = 1.0 / (units[uind])  # Unit in designmatrix
                if scale_by_F0:
                    un *= u.s
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
                # NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Update Uncertainties
            self.set_param_uncertainties(fitperrs)

            # Compute the noise realizations if possible
            if not full_cov:
                noise_dims = self.model.noise_model_dimensions(self.toas)
                noise_resids = {}
                for comp in noise_dims.keys():
                    p0 = noise_dims[comp][0] + ntmpar
                    p1 = p0 + noise_dims[comp][1]
                    noise_resids[comp] = np.dot(M[:, p0:p1], xhat[p0:p1]) * u.s
                self.resids.noise_resids = noise_resids

        # Update START/FINISH params
        self.model.START.value = self.toas.first_MJD
        self.model.FINISH.value = self.toas.last_MJD

        return chi2


class WidebandTOAFitter(Fitter):  # Is GLSFitter the best here?
    """ A class to for fitting TOAs and other independent measured data.

    Parameters
    ----------
    fit_data: data object or a tuple of data objects.
        The data to fit for. If one data are give, it will assume all the fit
        data set are packed in this one data object. If more than one data
        objects are provided, the size of 'fit_data' has to match the
        'fit_data_names'. In this fitter, the first fit data should be a TOAs object.
    model: a pint timing model instance
        The initial timing model for fitting.
    fit_data_names: list of str
        The names of the data fit for.
    additional_args: dict, optional
        The additional arguments for making residuals.
    """

    def __init__(
        self, fit_data, model, fit_data_names=["toa", "dm"], additional_args={}
    ):
        self.model_init = model
        # Check input data and data_type
        self.fit_data_names = fit_data_names
        # convert the non tuple input to a tuple
        if not isinstance(fit_data, (tuple, list)):
            fit_data = [fit_data]
        if not isinstance(fit_data[0], TOAs):
            raise ValueError("The first data set should be a TOAs object.")
        if len(fit_data_names) == 0:
            raise ValueError("Please specify the fit data.")
        if len(fit_data) > 1 and len(fit_data_names) != len(fit_data):
            raise ValueError(
                "If one more data sets are provided, the fit "
                "data have to match the fit data names."
            )
        self.fit_data = fit_data
        self.additional_args = additional_args
        # Get the makers for fitting parts.
        self.reset_model()
        self.resids_init = copy.deepcopy(self.resids)
        self.designmatrix_makers = []
        for data_resids in self.resids.residual_objs:
            self.designmatrix_makers.append(
                DesignMatrixMaker(data_resids.residual_type, data_resids.unit)
            )

        # Add noise design matrix maker
        self.noise_designmatrix_maker = DesignMatrixMaker("toa_noise", u.s)
        #
        self.covariancematrix_makers = []
        for data_resids in self.resids.residual_objs:
            self.covariancematrix_makers.append(
                CovarianceMatrixMaker(data_resids.residual_type, data_resids.unit)
            )

        self.method = "General_Data_Fitter"

    @property
    def toas(self):
        return self.fit_data[0]

    def make_combined_residuals(self, add_args={}):
        resid_obj = []
        if len(self.fit_data) == 1:
            for data_name in self.fit_data_names:
                r_obj = pr.Residuals(
                    self.fit_data[0],
                    self.model,
                    residual_type=data_name,
                    **add_args.get(data_name, {})
                )
                resid_obj.append(r_obj)
        else:
            for ii, data_name in enumerate(self.fit_data_names):
                r_obj = pr.Residuals(
                    self.fit_data[ii],
                    self.model,
                    residual_type=data_name,
                    **add_args.get(data_name, {})
                )
                resid_obj.append(r_obj)
        # Place the residual collector
        return pr.CombinedResiduals(resid_obj)

    def reset_model(self):
        """Reset the current model to the initial model."""
        self.model = copy.deepcopy(self.model_init)
        self.update_resids()
        self.fitresult = []

    def update_resids(self):
        """Update the residuals. Run after updating a model parameter."""
        self.resids = self.make_combined_residuals(self.additional_args)

    def get_designmatrix(self):
        design_matrixs = []
        fit_params = list(self.get_fitparams().keys())
        if len(self.fit_data) == 1:
            for ii, dmatrix_maker in enumerate(self.designmatrix_makers):
                design_matrixs.append(
                    dmatrix_maker(self.fit_data[0], self.model, fit_params, offset=True)
                )
        else:
            for ii, dmatrix_maker in enumerate(self.designmatrix_makers):
                design_matrixs.append(
                    dmatrix_maker(
                        self.fit_data[ii], self.model, fit_params, offset=True
                    )
                )
        return combine_design_matrices_by_quantity(design_matrixs)

    def get_noise_covariancematrix(self):
        # TODO This needs to be more general
        cov_matrixs = []
        if len(self.fit_data) == 1:
            for ii, cmatrix_maker in enumerate(self.covariancematrix_makers):
                cov_matrixs.append(cmatrix_maker(self.fit_data[0], self.model))
        else:
            for ii, cmatrix_maker in enumerate(self.covariancematrix_makers):
                cov_matrixs.append(cmatrix_maker(self.fit_data[ii], self.model))

        return combine_covariance_matrix(cov_matrixs)

    def get_data_uncertainty(self, data_name, data_obj):
        """ Get the data uncertainty from the data  object.

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

    def scaled_all_sigma(self,):
        """ Scale all data's uncertainty. If the function of scaled_`data`_sigma
        is not given. It will just return the original data uncertainty.
        """
        scaled_sigmas = []
        sigma_units = []
        for ii, fd_name in enumerate(self.fit_data_names):
            func_name = "scaled_{}_uncertainty".format(fd_name)
            sigma_units.append(self.resids.residual_objs[ii].unit)
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

    def fit_toas(self, maxiter=1, threshold=False, full_cov=False):
        # Maybe change the name to do_fit?
        # check that params of timing model have necessary components
        # self.model.maskPar_has_toas_check(self.toas)
        chi2 = 0
        for i in range(max(maxiter, 1)):
            fitp = self.get_fitparams()
            fitpv = self.get_fitparams_num()
            fitperrs = self.get_fitparams_uncertainty()

            # Define the linear system
            d_matrix = self.get_designmatrix()
            M, params, units, scale_by_F0 = (
                d_matrix.matrix,
                d_matrix.derivative_params,
                d_matrix.param_units,
                d_matrix.scaled_by_F0,
            )

        # Get residuals and TOA uncertainties in seconds
        if i == 0:
            self.update_resids()
        # Since the residuals may not have the same unit. Thus the residual here
        # has no unit.
        residuals = self.resids.resids

        # get any noise design matrices and weight vectors
        if not full_cov:
            # We assume the fit date type is toa
            Mn = self.noise_designmatrix_maker(self.toas, self.model)
            phi = self.model.noise_model_basis_weight(self.toas)
            phiinv = np.zeros(M.shape[1])
            if Mn is not None and phi is not None:
                phiinv = np.concatenate((phiinv, 1 / phi))
                new_d_matrix = combine_design_matrices_by_param(d_matrix, Mn)
                M, params, units, scale_by_F0 = (
                    new_d_matrix.matrix,
                    new_d_matrix.derivative_params,
                    new_d_matrix.param_units,
                    new_d_matrix.scaled_by_F0,
                )

        # normalize the design matrix
        norm = np.sqrt(np.sum(M ** 2, axis=0))
        ntmpar = len(fitp)
        if M.shape[1] > ntmpar:
            norm[ntmpar:] = 1
        if np.any(norm == 0):
            # Make this a LinAlgError so it looks like other bad matrixness
            raise sl.LinAlgError("One or more of the design-matrix columns is null.")
        M /= norm

        # compute covariance matrices
        if full_cov:
            cov = self.get_noise_covariancematrix().matrix
            cf = sl.cho_factor(cov)
            cm = sl.cho_solve(cf, M)
            mtcm = np.dot(M.T, cm)
            mtcy = np.dot(cm.T, residuals)

        else:
            Nvec = self.scaled_all_sigma() ** 2

            cinv = 1 / Nvec
            mtcm = np.dot(M.T, cinv[:, None] * M)
            mtcm += np.diag(phiinv)
            mtcy = np.dot(M.T, cinv * residuals)

        if maxiter > 0:
            try:
                c = sl.cho_factor(mtcm)
                xhat = sl.cho_solve(c, mtcy)
                xvar = sl.cho_solve(c, np.eye(len(mtcy)))
            except sl.LinAlgError:
                U, s, Vt = sl.svd(mtcm, full_matrices=False)

                if threshold:
                    threshold_val = np.finfo(np.longdouble).eps * max(M.shape) * s[0]
                    s[s < threshold_val] = 0.0

                xvar = np.dot(Vt.T / s, Vt)
                xhat = np.dot(Vt.T, np.dot(U.T, mtcy) / s)
            newres = residuals - np.dot(M, xhat)
            # compute linearized chisq
            if full_cov:
                chi2 = np.dot(newres, sl.cho_solve(cf, newres))
            else:
                chi2 = np.dot(newres, cinv * newres) + np.dot(xhat, phiinv * xhat)
        else:
            newres = residuals
            if full_cov:
                chi2 = np.dot(newres, sl.cho_solve(cf, newres))
            else:
                chi2 = np.dot(newres, cinv * newres)
            return chi2

        # compute absolute estimates, normalized errors, covariance matrix
        dpars = xhat / norm
        errs = np.sqrt(np.diag(xvar)) / norm
        covmat = (xvar / norm).T / norm
        self.covariance_matrix = covmat
        self.correlation_matrix = (covmat / errs).T / errs

        for ii, pn in enumerate(fitp.keys()):
            uind = params.index(pn)  # Index of designmatrix
            # Here we use design matrix's label, so the unit goes to normal.
            # instead of un = 1 / (units[uind])
            un = units[uind]
            if scale_by_F0:
                un *= u.s
            pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
            fitpv[pn] = np.longdouble((pv + dpv) / fitp[pn].units)
            # NOTE We need some way to use the parameter limits.
            fitperrs[pn] = errs[uind]
        self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
        # Update Uncertainties
        self.set_param_uncertainties(fitperrs)

        # Compute the noise realizations if possible
        if not full_cov:
            noise_dims = self.model.noise_model_dimensions(self.toas)
            noise_resids = {}
            for comp in noise_dims.keys():
                p0 = noise_dims[comp][0] + ntmpar
                p1 = p0 + noise_dims[comp][1]
                noise_resids[comp] = np.dot(M[:, p0:p1], xhat[p0:p1]) * u.s
            self.resids.noise_resids = noise_resids

        # Update START/FINISH params
        self.model.START.value = self.toas.first_MJD
        self.model.FINISH.value = self.toas.last_MJD

        return chi2
