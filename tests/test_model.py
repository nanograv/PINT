#! /usr/bin/env python
import time, sys
import pint.models as tm
from pint.phase import Phase
from pint import toa
import matplotlib.pyplot as plt
import numpy

m = tm.StandardTimingModel()
m.read_parfile('J1744-1134.basic.par')

print "model.as_parfile():"
print m.as_parfile()

try:
    planet_ephems = m.PLANET_SHAPIRO.value
except AttributeError:
    planet_ephems = False

sys.stderr.write("Reading TOAs...\n")
t0 = time.time()
t = toa.TOAs('J1744-1134.Rcvr1_2.GASP.8y.x.tim')
sys.stderr.write("Applying clock corrections...\n")
t.apply_clock_corrections()
sys.stderr.write("Computing observatory positions and velocities...\n")
t.compute_posvels(planets=planet_ephems)
time_toa = time.time() - t0
sys.stderr.write("Read/corrected TOAs in %.3f sec\n" % time_toa)

mjds = numpy.array([x.mjd.value for x in t.toas])
errs = t.get_errors()
resids = numpy.zeros_like(mjds)

sys.stderr.write("Computing residuals...\n")
t0 = time.time()
for ii, tt in enumerate(t.toas):
    p = m.phase(tt)
    resids[ii] = p.frac

time_phase = time.time() - t0
sys.stderr.write("Computed phases in %.3f sec\n" % time_phase)

plt.plot(mjds, resids, "x")
plt.xlabel("MJDs")
plt.ylabel("Residuals (phase)")
plt.show()
