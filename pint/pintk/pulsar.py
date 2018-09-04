'''
A wrapper around pulsar functions for pintkinter to use. This object will be shared
between widgets in the main frame and will contain the pre/post fit model, toas, 
pre/post fit residuals, and other useful information.
'''
from __future__ import print_function
from __future__ import division
import os, sys

# Numpy etc.
import numpy as np
import astropy.units as u
from astropy.time import Time
from enum import Enum

#Pint imports
import pint.models
import pint.toa
import pint.fitter
import pint.residuals
        
plot_labels = ['pre-fit', 'post-fit', 'mjd', 'year', 'orbital phase', 'serial', \
    'day of year', 'frequency', 'TOA error', 'rounded MJD']

# Some parameters we do not want to add a fitting checkbox for:
nofitboxpars = ['PSR', 'START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
    'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES', 'PLANET_SHAPIRO']

class Fitters(Enum):
    POWELL = 0
    WLS = 1
    GLS = 2

class Pulsar(object):
    '''
    Wrapper class for a pulsar. Contains the toas, model, residuals, and fitter
    '''

    def __init__(self, parfile=None, timfile=None):
        super(Pulsar, self).__init__()
        
        print('STARTING LOADING OF PULSAR %s' % str(parfile))
        
        if parfile is not None and timfile is not None:
            self.parfile = parfile
            self.timfile = timfile
        else:
            raise ValueError("No valid pulsar to load")

        self.prefit_model = pint.models.get_model(self.parfile)
        print("prefit_model.as_parfile():")
        print(self.prefit_model.as_parfile())

        try:
            planet_ephems = self.prefit_model.PLANET_SHAPIRO.value
        except AttributeError:
            planet_ephemes = False

        self.toas = pint.toa.get_TOAs(self.timfile)
        self.toas.print_summary()

        self.prefit_resids = pint.residuals.resids(self.toas, self.prefit_model)
        print("RMS PINT residuals are %.3f us\n" % \
              self.prefit_resids.time_resids.std().to(u.us).value)
        self.fitter = Fitters.WLS
        self.fitted = False

    @property
    def name(self):
        return getattr(self.prefit_model, 'PSR').value

    def __getitem__(self, key):
        try:
            return getattr(self.prefit_model, key)
        except AttributeError:
            print('Parameter %s was not found in pulsar model %s' % (key, self.name))
            return None

    def __contains__(self, key):
        return key in self.prefit_model.params
   
    def reset_model(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.update_resids()

    def reset_TOAs(self):
        self.toas = pint.toa.get_TOAs(self.timfile)
        self.update_resids()

    def resetAll(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.toas = pint.toa.get_TOAs(self.timfile)
        self.update_resids()
   
    def update_resids(self):
        self.prefit_resids = pint.residuals.resids(self.toas, self.prefit_model)
        if self.fitted:
            self.postfit_resids = pint.residuals.resids(self.toas, self.postfit_model)

    def orbitalphase(self):
        '''
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of zeros
        '''
        if 'PB' not in self:
            print("WARNING: This is not a binary pulsar")
            return np.zeros(len(self.toas))

        toas = self.toas.get_mjds()

        if 'T0' in self and not self['T0'].quantity is None:
            tpb = (toas.value - self['T0'].value) / self['PB'].value
        elif 'TASC' in self and not self['TASC'].quantity is None:
            tpb = (toas.value - self['TASC'].value) / self['PB'].value
        else:
            print("ERROR: Neither T0 nor TASC set")
            return np.zeros(len(toas))
        
        phase = np.modf(tpb)[0]
        phase[phase < 0] += 1
        return phase * u.cycle

    def dayofyear(self):
        '''
        Return the day of the year for all the TOAs of this pulsar
        '''
        t = Time(self.toas.get_mjds(), format='mjd')
        year = Time(np.floor(t.decimalyear), format='decimalyear')
        return (t.mjd - year.mjd) * u.day

    def year(self):
        '''
        Return the decimal year for all the TOAs of this pulsar
        '''
        t = Time(self.toas.get_mjds(), format='mjd')
        return (t.decimalyear) * u.year
    
    def write_fit_summary(self):
        '''
        Summarize fitting results
        '''
        if self.fitted:
            chi2 = self.postfit_resids.chi2
            wrms = np.sqrt(chi2 / self.toas.ntoas)
            print('Post-Fit Chi2:\t\t%.8g us^2' % chi2)
            print('Post-Fit Weighted RMS:\t%.8g us' % wrms)
            print('%19s\t%24s\t%24s\t%24s\t%24s' % 
                  ('Parameter', 'Pre-Fit', 'Post-Fit', 'Uncertainty', 'Difference'))
            print('-' * 144)
            fitparams = [p for p in self.prefit_model.params 
                         if not getattr(self.prefit_model, p).frozen]
            for key in fitparams:
                line = '%8s ' % key
                pre = getattr(self.prefit_model, key)
                post = getattr(self.postfit_model, key)
                line += '%10s\t' % ('' if post.units is None else str(post.units))
                if post.quantity is not None:
                    line += '%24s\t' % pre.print_quantity(pre.quantity)
                    line += '%24s\t' % post.print_quantity(post.quantity)
                    if post.uncertainty is not None:
                        line += '%24s \t' % post.print_uncertainty(post.uncertainty)
                    else:
                        line += '%24s \t' % ''
                    try:
                        line += '%24s' % (post.value - pre.value)
                    except:
                        pass
                print(line)
                #print('%8s %8s\t%16.10g\t%16.10g\t%16.8g\t%16.8g' % (key, units,
                #        preval, postval, unc, diff))
        else:
            print('Pulsar has not been fitted yet!')

    def fit(self, iters=1):
        '''
        Run a fit using the specified fitter
        '''
        if self.fitted:
            self.prefit_model = self.postfit_model
            self.prefit_resids = self.postfit_resids

        if self.fitter == Fitters.POWELL:
            fitter = pint.fitter.PowellFitter(self.toas, self.prefit_model)
        elif self.fitter == Fitters.WLS:
            fitter = pint.fitter.WlsFitter(self.toas, self.prefit_model)
        elif self.fitter == Fitters.GLS:
            fitter = pint.fitter.GLSFitter(self.toas, self.prefit_model)
        chi2 = self.prefit_resids.chi2
        wrms = np.sqrt(chi2 / self.toas.ntoas)
        print('Pre-Fit Chi2:\t\t%.8g us^2' % chi2)
        print('Pre-Fit Weighted RMS:\t%.8g us' % wrms)
        
        fitter.fit_toas(maxiter=1)
        self.postfit_model = fitter.model
        self.postfit_resids = pint.residuals.resids(self.toas, self.postfit_model)
        self.fitted = True
        self.write_fit_summary()
