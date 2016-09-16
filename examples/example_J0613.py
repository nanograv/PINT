"""Various tests to assess the performance of the J0623-0200."""
import pint.models.model_builder as mb
import pint.toa as toa
import libstempo as lt
import matplotlib.pyplot as plt
import tempo2_utils
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest

# Using Nanograv data J0623-0200
datadir = '../tests/datafile'
parfile = os.path.join(datadir, 'J0613-0200_NANOGrav_dfg+12_TAI_FB90.par')
timfile = os.path.join(datadir, 'J0613-0200_NANOGrav_dfg+12.tim')

# libstempo calculation
print "libstempo calculation"
psr = lt.tempopulsar(parfile, timfile)
# Build PINT model
print "PINT calculation"
m = mb.get_model(parfile)
# Get toas to pint
toas = toa.get_TOAs(timfile, planets=False, ephem='DE405')
tt = toas.table

t2_resids = psr.residuals()
presids_us = resids(toas, m).time_resids
# Plot residuals
plt.errorbar(toas.get_mjds(high_precision=False), presids_us.value,
            toas.get_errors(), fmt='x')
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.grid()
plt.show()
diff = (presids_us - t2_resids * u.second).to(u.us)
plt.plot(toas.get_mjds(high_precision=False), diff, '+')
plt.xlabel('Mjd (DAY)')
plt.ylabel('residule difference (us)')
plt.title('Residule difference between PINT and tempo2')
plt.show()
