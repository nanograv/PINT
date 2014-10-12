import time
import astropy.units as u
import pint.models as tm
from pint.phase import *

t0 = time.time()

if 0:
    from pint import toaTable
    tt = toaTable.TOAs("tests/J1744-1134.Rcvr1_2.GASP.8y.x.tim")
    tt.compute_planet_posvel_c(planets=True)
else:
    from pint import toa
    tt = toa.get_TOAs("tests/J1744-1134.Rcvr1_2.GASP.8y.x.tim")

m = tm.StandardTimingModel()
m.read_parfile('tests/J1744-1134.basic.par')

if 0:
    p = m.phase_table(tt)
else:
    p = numpy.asarray([m.phase(t) for t in tt.table])

t1 = time.time()

print t1 - t0



