#! /usr/bin/env python

import time, sys
import pint.models as tm
from pint.phase import Phase

m = tm.StandardTimingModel()
m.read_parfile('J1744-1134.basic.par')

print "model.as_parfile():"
print m.as_parfile()

from pint import toa
t0 = time.time()
t = toa.TOAs('J1744-1134.Rcvr1_2.GASP.8y.x.tim')
t.apply_clock_corrections()
t.compute_posvels()
time_toa = time.time() - t0
sys.stderr.write("Read/corrected TOAs in %.3f sec\n" % time_toa)

t0 = time.time()
for tt in t.toas:
    p = m.phase(tt)
    d = m.delay(tt)
    print tt.mjd.mjd, tt.freq, p.int, p.frac, d
time_phase = time.time() - t0
sys.stderr.write("Computed phases/delays in %.3f sec\n" % time_phase)
