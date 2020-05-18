from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy

import astropy.units as u
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
from astropy import log
import astropy.constants as const
import pint.utils
from pint.models.pulsar_binary import PulsarBinary
from pint import Tsun

from pint.residuals import Residuals
from pint.models.parameter import (
    AngleParameter,
    prefixParameter,
    strParameter,
    floatParameter,
)

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
            self.resids_init = Residuals(toas=toas, model=model)
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

    def update_resids(self, set_pulse_nums=False):
        """Update the residuals. Run after updating a model parameter."""
        self.resids = Residuals(
            toas=self.toas, model=self.model, set_pulse_nums=set_pulse_nums
        )

    def set_fitparams(self, *params):
        """Update the "frozen" attribute of model parameters.

        Ex. fitter.set_fitparams('F0','F1')
        """
        fit_params_name = []
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

        # Next, print the model parameters
        s += "{:<14s} {:^20s} {:^28s} {}\n".format("PAR", "Prefit", "Postfit", "Units")
        s += "{:<14s} {:>20s} {:>28s} {}\n".format(
            "=" * 14, "=" * 20, "=" * 28, "=" * 5
        )
        for pn in list(self.get_allparams().keys()):
            if nodmx and pn.startswith("DMX"):
                continue
            prefitpar = getattr(self.model_init, pn)
            par = getattr(self.model, pn)
            if par.value is not None:
                if isinstance(par, strParameter):
                    s += "{:14s} {:>20s} {:28s} {}\n".format(
                        pn, prefitpar.value, "", par.units
                    )
                elif isinstance(par, AngleParameter):
                    # Add special handling here to put uncertainty into arcsec
                    if par.frozen:
                        s += "{:14s} {:>20s} {:>28s} {} \n".format(
                            pn, str(prefitpar.quantity), "", par.units
                        )
                    else:
                        if par.units == u.hourangle:
                            uncertainty_unit = pint.hourangle_second
                        else:
                            uncertainty_unit = u.arcsec
                        s += "{:14s} {:>20s}  {:>16s} +/- {:.2g} \n".format(
                            pn,
                            str(prefitpar.quantity),
                            str(par.quantity),
                            par.uncertainty.to(uncertainty_unit),
                        )

                else:
                    # Assume a numerical parameter
                    if par.frozen:
                        s += "{:14s} {:20g} {:28s} {} \n".format(
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
                        s += "{:14s} {:20g} {:28SP} {} \n".format(
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
                    if not self.model.SINI.frozen:
                        si = ufloat(
                            self.model.SINI.quantity.value,
                            self.model.SINI.uncertainty.value,
                        )
                        s += "From SINI in model:\n"
                        s += "    cos(i) = {:SP}\n".format(um.sqrt(1 - si ** 2))
                        s += "    i = {:SP} deg\n".format(um.asin(si) * 180.0 / np.pi)

                    psrmass = pint.utils.pulsar_mass(
                        self.model.PB.quantity,
                        self.model.A1.quantity,
                        self.model.M2.quantity,
                        np.arcsin(self.model.SINI.quantity),
                    )
                    s += "Pulsar mass (Shapiro Delay) = {}".format(psrmass)

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


class PowellFitter(Fitter):
    """A class for Scipy Powell fitting method. This method searches over
       parameter space. It is a relative basic method.
    """

    def __init__(self, toas, model):
        super(PowellFitter, self).__init__(toas, model)
        self.method = "Powell"

    def fit_toas(self, maxiter=20):
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
            # Updata Uncertainties
            self.set_param_uncertainties(fitperrs)

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
                cov = self.model.covariance_matrix(self.toas)
                cf = sl.cho_factor(cov)
                cm = sl.cho_solve(cf, M)
                mtcm = np.dot(M.T, cm)
                mtcy = np.dot(cm.T, residuals)

            else:
                Nvec = self.model.scaled_sigma(self.toas).to(u.s).value ** 2
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

        return chi2
