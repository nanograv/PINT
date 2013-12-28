#! /usr/bin/env python

import pint.models as tm
from pint.phase import Phase

Model = tm.generate_timing_model("Model",(
    tm.Spindown,
    tm.Astrometry,
    tm.Dispersion))
m = Model()
m.read_parfile('J1744-1134.basic.par')

print "model.as_parfile():"
print m.as_parfile()

from pint import toa
t = toa.TOAs('J1744-1134.Rcvr1_2.GASP.8y.x.tim')
t.apply_clock_corrections()
t.compute_posvels()

for tt in t.toas:
    p = m.phase(tt)
    d = m.delay(tt)
    print tt.mjd.mjd, tt.freq, p.int, p.frac, d
