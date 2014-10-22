# fitter.py
# Defines the basic TOA fitter class
import numpy
import astropy.units as u
import astropy.coordinates.angles
import copy
import scipy.optimize as opt
from utils import has_astropy_unit

class resids(object):
    """resids(toa=None, model=None)"""

    def __init__(self, toas=None, model=None):
        self.toas = toas
        self.model = model
        if toas is not None and model is not None:
            self.phase_resids = self.calc_phase_resids()
            self.time_resids = self.calc_time_resids()
            self.chi2 = self.calc_chi2()
            self.dof = self.get_dof()
            self.chi2_reduced = self.chi2 / self.dof
        else:
            self.phase_resids = None
            self.time_resids = None

    def calc_phase_resids(self):
        """Return timing model residuals in pulse phase."""
        return self.model.phase(self.toas.table).frac

    def calc_time_resids(self):
        """Return timing model residuals in time (seconds)."""
        if self.phase_resids==None:
            self.phase_resids = self.calc_phase_resids()
        return (self.phase_resids / self.get_PSR_freq()).to(u.s)

    def get_PSR_freq(self):
        """Return pulsar rotational frequency in Hz. model.F0 must be defined."""
        if self.model.F0.units != 'Hz':
            ValueError('F0 units must be Hz')
        # All residuals require the model pulsar frequency to be defined
        F0names = ['F0', 'nu'] # recognized parameter names, needs to be changed
        nF0 = 0
        for n in F0names:
            if n in self.model.params:
                F0 = getattr(self.model, n).value
                nF0 += 1
        if nF0 == 0:
            raise ValueError('no PSR frequency parameter found; ' +
                             'valid names are %s' % F0names)
        if nF0 > 1:
            raise ValueError('more than one PSR frequency parameter found; ' +
                             'should be only one from %s' % F0names)
        return F0 * u.Hz

    def calc_chi2(self):
        """Return the weighted chi-squared for the model and toas."""
        # Residual units are in seconds. Error units are in microseconds.
        return ((self.time_resids / self.toas.get_errors()).decompose()**2.0).sum()

    def get_dof(self):
        """Return number of degrees of freedom for the model."""
        dof = self.toas.ntoas
        for p in self.model.params:
            dof -= bool(not getattr(self.model, p).frozen)
        return dof

    def get_reduced_chi2(self):
        """Return the weighted reduced chi-squared for the model and toas."""
        return self.calc_chi2() / self.get_dof()


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

    def get_fitparams(self):
        """Return a dict of param names and values for free parameters."""
        fitp = [p for p in self.model.params if not getattr(self.model,p).frozen]
        fitval = []
        for p in fitp:
            fitval.append(getattr(self.model,p).value)
        return {k: v for k,v in zip(fitp,fitval)}

    def set_params(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        Ex. fitter.set_params({'F0':60.1,'F1':-1.3e-15})
        """
        for p, val in zip(fitp.keys(), fitp.values()):
            # If value is unitless but model parameter is not, interpret the
            # value as being in the same units as the model
            modval=getattr(self.model,p).value
            # Right now while the code below preserves the unit, Angle types
            # become generic astropy quantities. Still, the timing model appears
            # to work.
            if not has_astropy_unit(val) and has_astropy_unit(modval):
                if type(modval) is astropy.coordinates.angles.Angle:
                    val=astropy.coordinates.angles.Angle(val, unit=modval.unit)
                else:
                    val=val*modval.unit
            getattr(self.model,p).value = val

    def minimize_func(self, x, *args):
        """Wrapper function for the residual class, meant to be passed to
        scipy.optimize.minimize. The function must take a single list of input
        values and a second optional tuple of input arguments, and return a
        quantitiy to be minimized (in this case chi^2).
        """
        # Minimze takes array of dimensionless input variables. Put back into
        # dict form and set the model.
        fitp = {k: v for k, v in zip(args, x)}
        self.set_params(fitp)
        # Get new residuals
        self.update_resids()
        # Return chi^2
        return self.resids.chi2

    def call_minimize(self, method='Powell', maxiter=20):
        """Wrapper to scipy.optimize.minimize function.
        Ex. fitter.call_minimize(method='Powell',maxiter=20)
        """
        # Initial guesses are model params
        fitp = self.get_fitparams()
        # Input variables must be unitless
        for k,v in zip(fitp.keys(),fitp.values()):
            if has_astropy_unit(v):
                fitp[k]=v.value
        self.fitresult=opt.minimize(self.minimize_func, fitp.values(),
                                    args=tuple(fitp.keys()),
                                    options={'maxiter':maxiter},
                                    method=method)
        # Update model and resids, as the last iteration of minimize is not
        # necessarily the one that yields the best fit
        self.minimize_func(numpy.atleast_1d(self.fitresult.x), *fitp.keys())

