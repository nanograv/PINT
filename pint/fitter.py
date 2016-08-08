# fitter.py
# Defines the basic TOA fitter class
import copy, numpy, numbers
import astropy.units as u
import astropy.coordinates.angles as ang
import scipy.optimize as opt, scipy.linalg as sl
from utils import has_astropy_unit
from .residuals import resids

class fitter(object):
    """fitter(toas=None, model=None)"""

    def __init__(self, toas=None, model=None):
        self.toas = toas
        self.model_init = model
        self.resids_init = resids(toas=toas, model=model)
        self.reset_model()

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
        for p in self.model.params:
            getattr(self.model,p).frozen = p not in params

    def get_allparams(self):
        """Return a dict of all param names and values."""
        return dict((k, getattr(self.model, k).value) for k in
                    self.model.params)

    def get_fitparams(self):
        """Return a dict of fittable param names and values."""
        return dict((k, getattr(self.model, k).value) for k in
                    self.model.params if not getattr(self.model, k).frozen)

    def get_fitparams_num(self):
        """Return a dict of fittable param names and numeric values."""
        return dict((k, getattr(self.model, k).value) for k in
                    self.model.params if not getattr(self.model, k).frozen)

    def set_params(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        Ex. fitter.set_params({'F0':60.1,'F1':-1.3e-15})
        """
        for k, v in fitp.items():
            # The check for astropy units should be able to go away once params are fixed
            getattr(self.model, k).num_value = v.value if has_astropy_unit(v) else v

    def minimize_func(self, x, *args):
        """Wrapper function for the residual class, meant to be passed to
        scipy.optimize.minimize. The function must take a single list of input
        values, x, and a second optional tuple of input arguments.  It returns
        a quantity to be minimized (in this case chi^2).
        """
        self.set_params({k: v for k, v in zip(args, x)})
        # Get new residuals
        self.update_resids()
        # Return chi^2
        return self.resids.chi2

    def call_minimize(self, method='Powell', maxiter=20):
        """Wrapper to scipy.optimize.minimize function.
        Ex. fitter.call_minimize(method='Powell',maxiter=20)
        """
        # Initial guesses are model params
        fitp = self.get_fitparams_num()
        self.fitresult=opt.minimize(self.minimize_func, fitp.values(),
                                    args=tuple(fitp.keys()),
                                    options={'maxiter':maxiter},
                                    method=method)
        # Update model and resids, as the last iteration of minimize is not
        # necessarily the one that yields the best fit
        self.minimize_func(numpy.atleast_1d(self.fitresult.x), *fitp.keys())

class wls_fitter(fitter):
    """fitter(toas=None, model=None)"""

    def __init__(self, toas=None, model=None):
        super(wls_fitter, self).__init__(toas=toas, model=model)

    def call_minimize(self, method='weighted', maxiter=20):
        """Run a linear weighted least-squared fitting method"""
        fitp = self.get_fitparams()

        # Input variables must be unitless
        for k, v in zip(fitp.keys(), fitp.values()):
            if has_astropy_unit(v):
                fitp[k] = v.value

        # Define the linear system
        M, params, units = self.model.designmatrix(toas=self.toas.table,
                incfrozen=False, incoffset=True)
        Nvec = numpy.array(self.toas.get_errors().to(u.s))**2
        self.update_resids()
        residuals = self.resids.time_resids.to(u.s)

        # Weighted linear fit
        Sigma_inv = numpy.dot(M.T / Nvec, M)
        U, s, Vt = sl.svd(Sigma_inv)
        Sigma = numpy.dot(Vt.T / s, U.T)
        dpars = numpy.dot(Sigma, numpy.dot(M.T, residuals.value / Nvec))

        # Uncertainties
        errs = numpy.sqrt(numpy.diag(Sigma))

        # Set the new parameter values
        # TODO: Now have to do the units manually, because not all parameters
        #       have units everywhere in the code yet. Eventually, this can be
        #       removed
        conv = {'F0': u.Hz, 'F1': u.Hz/u.s, 'RAJ':u.hourangle,
                'DECJ':u.degree, 'PMRA':u.mas/u.yr, 'PMDEC':u.mas/u.yr,
                'PX':u.mas, 'DM':u.s/u.s}

        # TODO: units and fitp have a different ordering. That is confusing
        for ii, pn in enumerate(fitp.keys()):
            uind = params.index(pn)             # Index of designmatrix
            un = 1.0 /  (units[uind]/u.s)       # Unit in designmatrix
            pv, dpv = fitp[pn] * conv[pn], dpars[uind] * un
            fitp[pn] = float( (pv+dpv) / conv[pn] )

        # TODO: Also record the uncertainties in minimize_func

        chi2 = self.minimize_func(list(fitp.values()), *fitp.keys())
