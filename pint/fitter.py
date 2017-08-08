import copy, numpy, numbers
import astropy.units as u
import abc
import scipy.optimize as opt, scipy.linalg as sl
from .residuals import resids


class Fitter(object):
    """ Base class for fitter.

    The fitting function should be defined as the fit_toas() method.

    Parameters
    ----------
    toas : a pint TOAs instance
        The input toas.
    model : a pint timing model instance
        The initial timing model for fitting.
    """
    def __init__(self, toas, model):
        self.toas = toas
        self.model_init = model
        self.resids_init = resids(toas=toas, model=model)
        self.reset_model()
        self.method = None

    def reset_model(self):
        """Reset the current model to the initial model."""
        self.model = copy.deepcopy(self.model_init)
        self.update_resids()
        self.fitresult = []

    def update_resids(self):
        """Update the residuals. Run after updating a model parameter."""
        self.resids = resids(toas=self.toas, model=self.model)

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
                if rn != '':
                    fit_params_name.append(rn)

        for p in self.model.params:
            getattr(self.model,p).frozen = p not in fit_params_name

    def get_allparams(self):
        """Return a dict of all param names and values."""
        return dict((k, getattr(self.model, k).quantity) for k in
                    self.model.params)

    def get_fitparams(self):
        """Return a dict of fittable param names and quantity."""
        return dict((k, getattr(self.model, k)) for k in
                    self.model.params if not getattr(self.model, k).frozen)

    def get_fitparams_num(self):
        """Return a dict of fittable param names and numeric values."""
        return dict((k, getattr(self.model, k).value) for k in
                    self.model.params if not getattr(self.model, k).frozen)

    def get_fitparams_uncertainty(self):
        return dict((k, getattr(self.model, k).uncertainty_value) for k in
                    self.model.params if not getattr(self.model, k).frozen)

    def set_params(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        Ex. fitter.set_params({'F0':60.1,'F1':-1.3e-15})
        """
        for k, v in fitp.items():
            getattr(self.model, k).value = v

    def set_param_uncertainties(self, fitp):
        for k, v in fitp.items():
            getattr(self.model, k).uncertainty_value = v

    def get_designmatrix(self):
        return self.model.designmatrix(toas=self.toas.table,
                incfrozen=False, incoffset=True)

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


class PowellFitter(Fitter):
    """A class for Scipy Powell fitting method. This method searches over
       parameter space. It is a relative basic method.
    """
    def __init__(self, toas, model):
        super(PowellFitter, self).__init__(toas, model)
        self.method = 'Powell'

    def fit_toas(self, maxiter=20):
        # Initial guesses are model params
        fitp = self.get_fitparams_num()
        self.fitresult=opt.minimize(self.minimize_func, list(fitp.values()),
                                    args=tuple(fitp.keys()),
                                    options={'maxiter':maxiter},
                                    method=self.method)
        # Update model and resids, as the last iteration of minimize is not
        # necessarily the one that yields the best fit
        self.minimize_func(numpy.atleast_1d(self.fitresult.x),
                           *list(fitp.keys()))


class WlsFitter(Fitter):
    """
       A class for weighted least square fitting method. The design matrix is
       required.
    """
    def __init__(self, toas=None, model=None):
        super(WlsFitter, self).__init__(toas=toas, model=model)
        self.method = 'weighted_least_square'

    def fit_toas(self, maxiter=1, thershold=False):
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
            Nvec = self.model.scaled_sigma(self.toas.table).to(u.s).value

            # "Whiten" design matrix and residuals by dividing by uncertainties
            M = M/Nvec.reshape((-1,1))
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
            #print 'log_10 cond=', numpy.log10(s.max()/s.min())
            # Note, Check the threshold from data precision level.Borrowed from
            # Numpy Curve fit.
            if thershold:
                threshold_val = numpy.finfo(numpy.longdouble).eps * max(M.shape) * s[0]
                s[s<threshold_val] = 0.0
            # Sigma = numpy.dot(Vt.T / s, U.T)
            # The post-fit parameter covariance matrix
            #   Sigma = V s^-2 V^T
            Sigma = numpy.dot(Vt.T / (s**2), Vt)
            # Parameter uncertainties.  Scale by fac recovers original units.
            errs = numpy.sqrt(numpy.diag(Sigma)) / fac

            # The delta-parameter values
            #   dpars = V s^-1 U^T r
            # Scaling by fac recovers original units
            dpars = numpy.dot(Vt.T, numpy.dot(U.T,residuals)/s) / fac
            for ii, pn in enumerate(fitp.keys()):
                uind = params.index(pn)             # Index of designmatrix
                un = 1.0 / (units[uind])     # Unit in designmatrix
                if scale_by_F0:
                    un *= u.s
                pv, dpv = fitpv[pn] * fitp[pn].units, dpars[uind] * un
                fitpv[pn] = numpy.longdouble((pv+dpv) / fitp[pn].units)
                #NOTE We need some way to use the parameter limits.
                fitperrs[pn] = errs[uind]
            chi2 = self.minimize_func(list(fitpv.values()), *list(fitp.keys()))
            # Updata Uncertainties
            self.set_param_uncertainties(fitperrs)

        return chi2
