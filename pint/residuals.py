import astropy.units as u
import numpy as np
from .phase import Phase

class resids(object):
    """resids(toa=None, model=None)"""

    def __init__(self, toas=None, model=None, weighted_mean=True):
        self.toas = toas
        self.model = model
        if toas is not None and model is not None:
            self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean)
            self.time_resids = self.calc_time_resids(weighted_mean=weighted_mean)
            self.chi2 = self.calc_chi2()
            self.dof = self.get_dof()
            self.chi2_reduced = self.chi2 / self.dof
        else:
            self.phase_resids = None
            self.time_resids = None

    def calc_phase_resids(self, weighted_mean=True):
        """Return timing model residuals in pulse phase."""
        rs = self.model.phase(self.toas.table)
        rs -= Phase(rs.int[0],rs.frac[0])
        if not weighted_mean:
            rs -= Phase(0.0,rs.frac.mean())
        else:
        # Errs for weighted sum.  Units don't matter since they will
        # cancel out in the weighted sum.
            w = 1.0/(np.array(self.toas.get_errors())**2)
            wm = (rs.frac*w).sum() / w.sum()
            rs -= Phase(0.0,wm)
        return rs.frac

    def calc_time_resids(self, weighted_mean=True):
        """Return timing model residuals in time (seconds)."""
        if self.phase_resids==None:
            self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean)
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
        if (self.toas.get_errors()==0.0).any():
            return np.inf
        else:
            # The self.time_resids is in the unit of "s", the error "us".
            # This is more correct way, but it is the slowest.
            #return (((self.time_resids / self.toas.get_errors()).decompose()**2.0).sum()).value

            # This method is faster then the method above but not the most correct way
            #return ((self.time_resids.to(u.s) / self.toas.get_errors().to(u.s)).value**2.0).sum()

            # This the fastest way, but highly depend on the assumption of time_resids and
            # error units.
            return ((self.time_resids.value * 1e6 / self.toas.get_errors())**2.0).sum()
    def get_dof(self):
        """Return number of degrees of freedom for the model."""
        dof = self.toas.ntoas
        for p in self.model.params:
            dof -= bool(not getattr(self.model, p).frozen)
        return dof

    def get_reduced_chi2(self):
        """Return the weighted reduced chi-squared for the model and toas."""
        return self.calc_chi2() / self.get_dof()
