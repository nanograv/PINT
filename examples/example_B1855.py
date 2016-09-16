"""Various tests to assess the performance of the B1855+09."""
import pint.models.model_builder as mb
import pint.toa as toa
import libstempo as lt
import matplotlib.pyplot as plt
import tempo2_utils
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest

# Using Nanograv data B1855
datadir = '../tests/datafile'
parfile = os.path.join(datadir, 'B1855+09_NANOGrav_dfg+12_TAI_FB90.par')
timfile = os.path.join(datadir, 'B1855+09_NANOGrav_dfg+12.tim')

# libstempo calculation
print "libstempo calculation"
psr = lt.tempopulsar(parfile, timfile)
# Build PINT model
print "PINT calculation"
mdd = mb.get_model(parfile)
# Get toas to pint
toas = toa.get_TOAs(timfile, planets=False, ephem='DE405')
tt = toas.table
# Get residuals
t2_resids = psr.residuals()
presids_us = resids(toas, mdd).time_resids
# Plot residuals
plt.errorbar(toas.get_mjds(high_precision=False), presids_us.value,
            toas.get_errors(), fmt='x')
plt.title("%s Pre-Fit Timing Residuals" % mdd.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.grid()
plt.show()
# Plot tempo difference
diff = (presids_us - t2_resids * u.second).to(u.us)
plt.plot(toas.get_mjds(high_precision=False), diff, '+')
plt.xlabel('Mjd (DAY)')
plt.ylabel('residule difference (us)')
plt.title('Residule difference between PINT and tempo2')
plt.show()
