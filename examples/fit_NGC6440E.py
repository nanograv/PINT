#! /usr/bin/env python
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import matplotlib.pyplot as plt
import astropy.units as u
import os, sys

datadir = os.path.dirname(os.path.abspath(__file__))
parfile = os.path.join(datadir, 'NGC6440E.par')
timfile = os.path.join(datadir, 'NGC6440E.tim')

# Define the timing model
m = pint.models.StandardTimingModel()
m.read_parfile(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)

# These are pre-fit residuals
rs = pint.residuals.resids(t, m).phase_resids
xt = t.get_mjds()
plt.plot(xt, rs, 'x')
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (phase)')
plt.grid()
plt.show()

# Now do the fit
print "Fitting..."
f = pint.fitter.fitter(t, m)
f.call_minimize()

# Print some basic params
print "Best fit has reduced chi^2 of", f.resids.chi2_reduced
print "RMS in phase is", f.resids.phase_resids.std()
print "RMS in time is", f.resids.time_resids.std().to(u.us)
print "\n Best model is:"
print f.model.as_parfile()

plt.errorbar(xt,
             f.resids.time_resids.to(u.us).value,
             t.get_errors().to(u.us).value, fmt='x')
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.grid()
plt.show()
