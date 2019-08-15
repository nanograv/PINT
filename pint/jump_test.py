#! /usr/bin/env python
from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.models.model_builder as mb
from pint.phase import Phase
from pint.utils import make_toas
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt

import astropy.units as u
import os

datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'J1833-38.par.ell1good')
timfile = os.path.join(datadir, 'j_test.tim')

# Define the timing model
m = mb.get_model(parfile)

print('A1',m.A1)
print('PB',m.PB)
exit()
# Read in the TOAs
t = pint.toa.get_TOAs(timfile)

print(type(t.table['flags']))
s = deepcopy(t)
table = deepcopy(t.table)
s.table = table
t.table['flags'][0]['testing']=123
print(s == t)
print(s.table['flags'])
print(t.table['flags'])
print(s.table == t.table)


# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.resids(t, m).phase_resids
xt = t.get_mjds()
#plt.plot(xt, rs, 'x', label = 'pre-fit')
#plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
#plt.xlabel('MJD')
#plt.ylabel('Residual (phase)')
#plt.grid()
#plt.show()

#convert jump flags to params

m.jump_flags_to_params(t)

# Now do the fit
print("Fitting...")
f = pint.fitter.WlsFitter(t, m)
print("BEFORE:",f.get_fitparams())
print(f.fit_toas())

print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())
print('f0',f.model.F0)
#print(t.table['flags'])
#print(m.JUMP1.toa_selector.select_result)
print('phase_deriv_funcs',f.model.phase_deriv_funcs)

#plot post fit residuals with error bars
plt.errorbar(xt.value,
             pint.residuals.resids(t, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
             t.get_errors().to(u.us).value, fmt='x', label = 'post-fit')
plt.plot(t.get_mjds(), pint.residuals.resids(t,m).time_resids.to(u.us).value, 'x', label = 'pre-fit')
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.legend()
plt.grid()
plt.show()
