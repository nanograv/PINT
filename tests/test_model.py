#! /usr/bin/env python
import pint.models as tm
from pint.phase import Phase
import matplotlib.pyplot as plt
import numpy

Model = tm.generate_timing_model("Model", \
                                 (tm.Spindown,
                                  tm.Astrometry,
                                  tm.Dispersion,
                                  tm.SolarSystemShapiro))
m = Model()
m.read_parfile('J1744-1134.basic.par')

print "model.as_parfile():"
print m.as_parfile()

from pint import toa
print "Reading TOAs..."
t = toa.TOAs('J1744-1134.Rcvr1_2.GASP.8y.x.tim')
print "Applying clock corrections..."
t.apply_clock_corrections()
print "Computing observatory positions and velocities..."
t.compute_posvels()

mjds = numpy.array([x.mjd.value for x in t.toas])
errs = t.get_errors()
resids = numpy.zeros_like(mjds)

print "Computing residuals..."
for ii, tt in enumerate(t.toas):
    p = m.phase(tt)
    d = m.delay(tt)
    resids[ii] = p.frac
    print tt.mjd.mjd, tt.freq, p.int, p.frac, d

plt.plot(mjds, resids, "x")
plt.xlabel("MJDs")
plt.ylabel("Residuals (phase)")
plt.show()
