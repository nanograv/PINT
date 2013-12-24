#! /usr/bin/env python

import timing_model as tm
from astrometry import Astrometry
from spindown import Spindown
from dispersion import Dispersion
from phase import Phase

Model = tm.generate_timing_model("Model",(Spindown,Astrometry,Dispersion))
m = Model()
m.read_parfile('J1744-1134.basic.par')

print "model.as_parfile():"
print m.as_parfile()

import toa
t = toa.TOAs('J1744-1134.Rcvr1_2.GASP.8y.x.tim')
t.apply_clock_corrections()
t.compute_posvels()

for tt in t.toas:
    p = m.phase(tt)
    d = m.delay(tt)
    print tt.mjd.mjd, tt.freq, p.int, p.frac, d
