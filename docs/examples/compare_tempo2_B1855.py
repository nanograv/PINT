"""Various tests to assess the performance of the B1855+09."""

# This example requires `tempo2_utils`. It is available at
# https://github.com/demorest/tempo_utils.

import pint.models.model_builder as mb
import pint.toa as toa
import pint.logging

# setup logging
pint.logging.setup(level="INFO")

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import astropy.units as u
from pint.residuals import Residuals as resids
import os
from astropy import log

try:
    import tempo2_utils
except ImportError:
    log.error(
        "This example requires tempo_utils, download from: https://github.com/demorest/tempo_utils and 'pip install .'"
    )
    raise

# Using Nanograv data B1855
import pint.config

parfile = pint.config.examplefile("B1855+09_NANOGrav_dfg+12_TAI_FB90.par")
timfile = pint.config.examplefile("B1855+09_NANOGrav_dfg+12.tim")

# libstempo calculation
print("tempo2 calculation")
tempo2_vals = tempo2_utils.general2(parfile, timfile, ["pre"])
# Build PINT model
print("PINT calculation")
mdd = mb.get_model(parfile)
# Get toas to pint
toas = toa.get_TOAs(timfile, planets=False, ephem="DE405", include_bipm=False)
# Get residuals
t2_resids = tempo2_vals["pre"]
presids_us = resids(toas, mdd).time_resids.to(u.us)
# Plot residuals
plt.errorbar(toas.get_mjds().value, presids_us.value, toas.get_errors().value, fmt="x")
print(toas.get_errors().value)
print(toas.get_mjds())

print(presids_us.value)
plt.title("%s Pre-Fit Timing Residuals" % mdd.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
plt.show()
# Plot tempo difference
diff = (presids_us - t2_resids * u.second).to(u.us)
plt.plot(toas.get_mjds(high_precision=False), diff, "+")
plt.xlabel("Mjd (DAY)")
plt.ylabel("residual difference (us)")
plt.title("Residual difference between PINT and tempo2")
plt.show()
