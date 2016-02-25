"""Various tests to assess the performance of the DD model."""
import pint.models.model_builder as mb
import pint.toa as toa
import libstempo as lt
import matplotlib.pyplot as plt
import tempo2_utils
import astropy.units as u
from pint.residuals import resids
import numpy as np
import os, unittest

datapath = os.path.join(os.environ['PINT'],'tests','datafile')
# Using Nanograv data B1855
parfile = os.path.join(datapath, 'B1855+09_NANOGrav_dfg+12_modified.par')
timfile = os.path.join(datapath, 'B1855+09_NANOGrav_dfg+12.tim')
# libstempo calculation
print "libstempo calculation"
psr = lt.tempopulsar(parfile, timfile)
# Build PINT model
print "PINT calculation"
mdd = mb.get_model(parfile)
# Get toas to pint
toas = toa.get_TOAs(timfile, planets=True)
tt = toas.table
# Run tempo2 general2 pluging
tempo2_vals = tempo2_utils.general2(parfile, timfile,
                                    ['tt2tb', 'roemer', 'post_phase',
                                     'shapiro', 'shapiroJ','bat','clock0',
                                     'clock1','clock2','clock3','clock4','sat',
                                     'tropo'])
# compute residules
t2_resids = tempo2_vals['post_phase'] / float(mdd.F0.num_value) * 1e6 * u.us
presids_us = resids(toas, mdd).time_resids.to(u.us)
toas = psr.toas()
toas.sort()
plt.plot(toas,presids_us-t2_resids)
plt.xlabel('Mjd (DAY)')
plt.ylabel('residule (us)')
plt.title('Residule difference between PINT and tempo2')
plt.show()
