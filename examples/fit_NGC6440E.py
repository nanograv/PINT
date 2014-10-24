#! /usr/bin/env python
import numpy as np
import pint.toa
import pint.models
import pint.fitter
import matplotlib.pyplot as plt

parfile = 'examples/NGC6440E.par'
timfile = 'examples/NGC6440E.tim'

# Define the timing model
m = pint.models.StandardTimingModel()
m.read_parfile(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)

# Now do the fit
f = pint.fitter.fitter(t, m)
f.call_minimize()

#plt.errorbar(t.get_MJDs(), f.residuals.time_residuals(),
#             t.get_errors(), fmt=None)
#plt.title("%s timing" % m.)
#plt.xlabel('MJD')
#plt.ylabel('Residual (us)')
#plt.grid()
#plt.show()
