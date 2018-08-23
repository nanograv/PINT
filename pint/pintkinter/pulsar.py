from __future__ import print_function
from __future__ import division
import os, sys

# Numpy etc.
import numpy as np
import time
import tempfile

# For date conversions
import astropy.units as u
from astropy.time import Time

import pint.models as pm
from pint.phase import Phase
from pint import toa
import pint.fitter
import pint.residuals
        
isolated = ['pre-fit', 'post-fit', 'mjd', 'year', 'serial', \
    'day of year', 'frequency', 'TOA error', 'elevation', \
    'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']
binary = ['orbital phase']

# Plot labels = isolated + binary
plot_labels = ['pre-fit', 'post-fit', 'mjd', 'year', 'orbital phase', 'serial', \
    'day of year', 'frequency', 'TOA error', 'elevation', \
    'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']

# Some parameters we do not want to add a fitting checkbox for:
nofitboxpars = ['PSR', 'START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
    'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES']

class Pulsar(object):
    '''
    Wrapper class for a pulsar. Contains the toas, model, residuals, and fitter
    '''

    def __init__(self, parfile=None, timfile=None):
        super(Pulsar, self).__init__()
        
        print('STARTING LOADING OF PULSAR %s' % str(parfile))
        
        if parfile is not None and timfile is not None:
            parfilename = parfile
            timfilename = timfile
        else:
            raise ValueError("No valid pulsar to load")

        self._model = pm.get_model(parfilename)
        print("model.as_parfile():")
        print(self._model.as_parfile())

        self.generate_fitparams()

        try:
            planet_ephems = self._model.PLANET_SHAPIRO.value
        except AttributeError:
            planet_ephemes = False

        self._toas = toa.get_TOAs(timfilename)
        self._toas.print_summary()

        self._resids = pint.residuals.resids(self._toas, self._model)
        self._prefit_resids = self._resids.time_resids
        print("RMS PINT residuals are %.3f us\n" % \
              self._prefit_resids.std().to(u.us).value)
        self._fitter = pint.fitter.WlsFitter(self._toas, self._model)

    @property
    def name(self):
        return getattr(self._model, 'PSR').value

    def __getitem__(self, key):
        try:
            return getattr(self._model, key)
        except AttributeError:
            print('Parameter %s was not found in pulsar model %s' % (key, self.name))
            return None

    def __contains__(self, key):
        return key in self._model.params
    
    @property
    def params(self):
        '''Returns a tuple of names of parameters in the model'''
        return self._model.params

    @property
    def fitparams(self):
        '''Returns a tuple of names of parameters that are fitted'''
        return self._fitpars

    @property
    def setparams(self):
        '''Returns a tuple of names of parameters that are set in the parfile'''
        return [p for p in self._model.params \
                if not getattr(self._model, p).quantity is None] 

    @property
    def compsSetParamsDict(self):
        '''Returns a dict of names of components and parameters that are set in the parfile'''
        ret = {}
        for comp in self._model.components:
            ret[comp] = [p for p in self._model.components[comp].params \
                         if not getattr(self._model, p).quantity is None]
        return ret

    @property
    def vals(self): 
        '''Returns a tuple of parameter values in the model'''
        return np.array([getattr(self._model, p).value for p in self.params])

    @vals.setter
    def vals(self, values):
        for key, val in zip(self.params, values):
            getattr(self._model, key).value = val

    @property
    def fitvals(self):
        '''Returns a vector of values of all fitted parameters'''
        return self._fitvals

    @fitvals.setter
    def fitvals(self, values):
        for key, val in zip(self.fitparams, values):
            getattr(self._model, key).value = val
        self._fitvals = values

    @property
    def setvals(self):
        '''Returns a vector of set values in the parfile'''
        return np.array([getattr(self._model, p).value for p in self.setparams])

    @setvals.setter
    def setvals(self, values):
        for key, val in zip(self.setparams, values):
            getattr(self._model, key).value = val

    @property
    def errs(self):
        '''Returns a vector of errors of all model parameters'''
        return np.array([getattr(self._model, p).uncertainty_value for p in self.params])

    @errs.setter
    def errs(self, values):
        for key, err in zip(self.params, values):
            getattr(self._model, key).uncertainty_value = err

    @property
    def fiterrs(self):
        '''Returns a vector of errors of fitted parameters'''
        return self._fiterrs

    @fiterrs.setter
    def fiterrs(self, values):
        for key, err in zip(self.fitparams, values):
            getattr(self._model, key).uncertainty_value = err
        self._fiterrs = values
        
    @property
    def seterrs(self, values):
        return np.array([getattr(self._model, p).uncertainty_value for p in self.setparams])

    @property
    def ndim(self):
        return len(self._fitpars)

    @property
    def deleted(self):
        return np.zeros(self._toas.ntoas, dtype=np.bool)

    @deleted.setter
    def deleted(self, values):
        pass

    @property
    def toas(self):
        '''Barycentric arrival times'''
        return np.array([t.value for t in self._toas.table['tdb']]) * u.d

    @property
    def stoas(self):
        '''Site arrival times'''
        return self._toas.get_mjds()

    @property
    def toaerrs(self):
        '''TOA uncertainties'''
        return self._toas.get_errors()

    @property
    def freqs(self):
        '''Observing frequencies'''
        return self._toas.table['freq'].quantity

    @property
    def residuals(self, updatebats=True, formresiduals=True):
        return self._fitter.resids.time_resids

    @property
    def prefitresiduals(self):
        return self._prefit_resids
    
    @property
    def chisq(self):
        return self._fitter.resids.chi2.value

    @property
    def orbitalphase(self):
        '''
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of zeros
        '''
        if 'PB' not in self:
            print("WARNING: This is not a binary pulsar")
            return np.zeros(len(self.toas))

        if 'T0' in self and not self['T0'].quantity is None:
            tpb = (self.toas.value - self['T0'].value) / self['PB'].value
        elif 'TASC' in self and not self['TASC'].quantity is None:
            tpb = (self.toas.value - self['TASC'].value) / self['PB'].value
        else:
            print("ERROR: Neither T0 nor TASC set")
            return np.zeros(len(self.toas))
        
        phase = np.modf(tpb)[0]
        phase[phase < 0] += 1
        return phase * u.cycle

    @property
    def dayofyear(self):
        '''
        Return the day of the year for all the TOAs of this pulsar
        '''
        t = Time(self.stoas, format='mjd')
        year = Time(np.floor(t.decimalyear), format='decimalyear')
        return (t.mjd - year.mjd) * u.day

    @property
    def year(self):
        t = Time(self.stoas, format='mjd')
        return (t.decimalyear) * u.year
        
    @property
    def siderealt(self):
        pass

    def generate_fitparams(self):
        self._fitpars = [p for p in self._model.params if not getattr(self._model, p).frozen]
        self._fitvals = np.zeros(len(self._fitpars))
        self._fiterrs = np.zeros(len(self._fitpars))
        for i in range(len(self._fitpars)):
            self._fitvals[i] = getattr(self._model, self._fitpars[i]).value
            self._fiterrs[i] = getattr(self._model, self._fitpars[i]).uncertainty_value

    def set_fit_state(self, parchanged, newstate):
        getattr(self._model, parchanged).frozen = not newstate
        self.generate_fitparams()

    def designmatrix(self, updatebats=True, fixunits=False):
        raise NotImplementedError
        return None
    
    def write_fit_summary(self):
        wrms = np.sqrt(self.chisq / self._toas.ntoas)
        print('Post-Fit Chi2:\t\t%.8g us^2' % self.chisq)
        print('Post-Fit Weighted RMS:\t%.8g us' % wrms)
        print('%17s\t%16s\t%16s\t%16s\t%16s' % 
              ('Parameter', 'Pre-Fit', 'Post-Fit', 'Uncertainty', 'Difference'))
        print('-' * 112)
        for key in self.fitparams:
            post = getattr(self._fitter.model, key).quantity
            units = post.unit
            pre = getattr(self._model, key).quantity.to(units)
            unc = getattr(self._fitter.model, key).uncertainty.to(units)
            diff = (post - pre).to(units)
            print('%8s %8s\t%16.10g\t%16.10g\t%16.8g\t%16.8g' % (key, units,
                                                             pre.value, 
                                                             post.value,
                                                             unc.value,
                                                             diff.value))

    def fit(self, iters=1):
        #Re-initialize fitter (start fit from scratch)
        self._fitter = pint.fitter.WlsFitter(self._toas, self._model)
        wrms = np.sqrt(self.chisq / self._toas.ntoas)
        print('Pre-Fit Chi2:\t\t%.8g us^2' % self.chisq)
        print('Pre-Fit Weighted RMS:\t%.8g us' % wrms)
        self._fitter.fit_toas(maxiter=1)
        self.write_fit_summary()
    
    def rd_hms(self):
        raise NotImplemented("Not done")
        #return self._psr.rd_hms()
        return None

    def savepar(self, parfile):
        #self._psr.savepar(parfile)
        pass

    def savetim(self, timfile):
        #self._psr(timfile)
        pass

    def phasejumps(self):
        return []

    def add_phasejump(self, mjd, phasejump):
        pass

    def remove_phasejumps(self):
        pass

    @property
    def nphasejumps(self):
        return self._psr.nphasejumps

    def data_from_label(self, label):
        """
        Given a label, return the data that corresponds to it

        @param label:   The label of which we want to obtain the data

        @return:    data, error, plotlabel
        """
        data, error, plotlabel = None, None, None

        if label == 'pre-fit':
            data = self.prefitresiduals.to(u.us)
            error = self.toaerrs.to(u.us)
            plotlabel = r"Pre-fit residual ($\mu$s)"
        elif label == 'post-fit':
            data = self.residuals.to(u.us)
            error = self.toaerrs.to(u.us)
            plotlabel = r"Post-fit residual ($\mu$s)"
        elif label == 'mjd':
            data = self.stoas
            error = self.toaerrs.to(u.d)
            plotlabel = r'MJD'
        elif label == 'orbital phase':
            data = self.orbitalphase
            error = None
            plotlabel = 'Orbital Phase'
        elif label == 'serial':
            data = np.arange(len(self.stoas)) * u.m / u.m
            error = None
            plotlabel = 'TOA number'
        elif label == 'day of year':
            data = self.dayofyear
            error = None
            plotlabel = 'Day of the year'
        elif label == 'year':
            data = self.year
            error = None
            plotlabel = 'Year'
        elif label == 'frequency':
            data = self.freqs
            error = None
            plotlabel = r"Observing frequency (MHz)"
        elif label == 'TOA error':
            data = self.toaerrs.to(u.us)
            error = None
            plotlabel = "TOA uncertainty"
        elif label == 'elevation':
            data = self.elevation
            error = None
            plotlabel = 'Elevation'
        elif label == 'rounded MJD':
            # TODO: Do we floor, or round like this?
            data = np.floor(self.stoas + 0.5 * u.d)
            error = self.toaerrs.to(u.d)
            plotlabel = r'MJD'
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        
        return data, error, plotlabel

    def mask(self, mtype='plot', flagID=None, flagVal=None):
        """
        Returns a mask of TOAs, depending on what is requestion by mtype

        @param mtype:   What kind of mask is requested. (plot, deleted, range)
        @param flagID:  If set, only give mask for a given flag (+flagVal)
        @param flagVal: If set, only give mask for a given flag (+flagID)
        """
        if mtype == 'deleted':
            msk = self.deleted
        elif mtype in ['range', 'plot']:
            if mtype == 'range':
                msk = np.ones(len(self.stoas), dtype=np.bool)
            elif mtype == 'plot':
                msk = np.logical_not(self.deleted)
            if 'START' in self.fitparams:
                msk[self.stoas < self['START'].quantity] = False
            if 'FINISH' in self.fitparams:
                msk[self.stoas > self['FINISH'].quantity] = False
        elif mtype=='noplot':
            msk = self.deleted
            if 'START' in self.fitparams:
                msk[self.stoas < self['START'].quantity] = True
            if 'FINISH' in self.fitparams:
                msk[self.stoas > self['FINISH'].val] = True
        
        return msk
