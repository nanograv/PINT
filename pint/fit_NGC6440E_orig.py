#! /usr/bin/env python
from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.random_models
import pint.models.model_builder as mb
from pint.phase import Phase
from pint.utils import make_toas, show_cov_matrix
import numpy as np
from copy import deepcopy
from collections import OrderedDict

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import astropy.units as u
import os

redge = 3
ledge = 0.5
datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'NGC6440E.par.orig')
timfile = os.path.join(datadir, 'NGC6440E.tim')

# Define the timing model
m = mb.get_model(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)
t0 = pint.toa.get_TOAs(timfile)
# Examples of how to select some subsets of TOAs
# These can be un-done using t.unselect()
#
# Use every other TOA
# t.select(np.where(np.arange(t.ntoas) % 2))

# Use only TOAs with errors < 30 us
# t.select(t.get_errors() < 30 * u.us)

# Use only TOAs from the GBT (although this is all of them for this example)
# t.select(t.get_obss() == 'gbt')

name = 'testing'
save = True
iter = 10
tmin = 53670
t.select(t.get_mjds() > tmin * u.d)#t = fit
t.select(t.get_mjds() < 53687 * u.d)
t0.select(t0.get_mjds() > 53600 * u.d)#t0 = graph
t0.select(t0.get_mjds() < 53750 * u.d)
# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.resids(t0, m).phase_resids
xt = t0.get_mjds()
plt.plot(xt, rs, 'x', label = 'pre-fit')
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (phase)')
plt.grid()
plt.show()

# Now do the fit
print("Fitting...")
f = pint.fitter.WlsFitter(t, m)
print(f.fit_toas())

#this is a complete graphing thing, just because of how the plot shows up when pre and post fit are shown
q = list(t0.get_mjds())
index = q.index([i for i in t0.get_mjds() if i > tmin*u.d][0])
rs_mean = pint.residuals.resids(t0,f.model).phase_resids[index:index+len(t.get_mjds())].mean()

params = f.get_fitparams_num()
# Print some basic params
#remove first row and column of cov matrix
ucov_mat = (((f.resids.unscaled_cov_matrix[0][1:]).T)[1:]).T
show_cov_matrix(ucov_mat,params.keys(),"Unscaled Cov Matrix",switchRD=True)
show_cov_matrix((((f.resids.scaled_cov_matrix[0][1:]).T)[1:]).T,params.keys(),"Scaled Cov Matrix",switchRD=True)
print("Mean vector is", params.values())
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())
print('-'*100)
if save:
    j = open(name+'.par','w')
    j.write(f.model.as_parfile())
    j.close()

#create and plot random models    
fake_toas, rss = pint.random_models.random(f, rs_mean, ledge_multiplier=ledge, redge_multiplier=redge, iter=iter)
for rs in rss:
    plt.plot(fake_toas, rs, '-k', alpha =0.3)
    
#plot post fit residuals with error bars
plt.errorbar(xt.value,
             pint.residuals.resids(t0, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
             t0.get_errors().to(u.us).value, fmt='x', label = 'post-fit')
plt.plot(t.get_mjds(), pint.residuals.resids(t,m).time_resids.to(u.us).value, 'x', label = 'fit points')
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.legend()
plt.grid()
plt.show()
