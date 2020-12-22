#! /usr/bin/env python
"""Demonstrate use of pint in a script."""
from __future__ import print_function, division
import pint.toa
import pint.fitter
import pint.residuals
import pint.models as mm

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import astropy.units as u
import os

datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")

# Read the timing model and the TOAs
m, t = mm.get_model_and_toas(parfile, timfile)

# If we wanted to do things separately we could do
# Define the timing model
# m = mm.get_model(parfile)
# Read in the TOAs, overriding some things from the model
# t = pint.toa.get_TOAs(timfile, model=m)

# Examples of how to select some subsets of TOAs
# These can be un-done using t.unselect()
#
# Use every other TOA
# t.select(np.where(np.arange(t.ntoas) % 2))

# Use only TOAs with errors < 30 us
# t.select(t.get_errors() < 30 * u.us)

# Use only TOAs from the GBT (although this is all of them for this example)
# t.select(t.get_obss() == 'gbt')

# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.Residuals(t, m).phase_resids
xt = t.get_mjds()
plt.plot(xt, rs, "x")
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()
plt.show()

# Now do the fit
print("Fitting.")
f = pint.fitter.WLSFitter(t, m)
# f = pint.fitter.PowellFitter(t, m)
f.fit_toas()
# f = pint.fitter.GLSFitter(t, m)
# f.fit_toas(full_cov=True)

# Print some basic params
print("Best fit has reduced chi^2 of", f.resids.reduced_chi2)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))

# Show the parameter correlation matrix
corm = f.get_correlation_matrix(pretty_print=True)

f.print_summary()

plt.errorbar(
    xt.value,
    f.resids.time_resids.to(u.us).value,
    t.get_errors().to(u.us).value,
    fmt="x",
)
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
plt.show()
