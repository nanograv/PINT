#! /usr/bin/env python
import time, sys, os
import pint.models as tm
from pint.phase import Phase
from pint import toa
import matplotlib.pyplot as plt
import numpy
import tempo2_utils

parfile = 'J1744-1134.basic2.par'
t1_parfile = 'J1744-1134.t1.par'
timfile = 'J1744-1134.Rcvr1_2.GASP.8y.x.tim'

m = tm.StandardTimingModel()
m.read_parfile(parfile)

print "model.as_parfile():"
print m.as_parfile()

try:
    planet_ephems = m.PLANET_SHAPIRO.value
except AttributeError:
    planet_ephems = False

sys.stderr.write("Reading TOAs...\n")
t0 = time.time()
t = toa.get_TOAs(timfile,ephem='DE405')
time_toa = time.time() - t0
sys.stderr.write("Read/corrected TOAs in %.3f sec\n" % time_toa)

mjds = numpy.array([x.mjd.value for x in t.toas])
d_tdbs = numpy.array([x.mjd.tdb.delta_tdb_tt[0] for x in t.toas])
errs = t.get_errors()
resids = numpy.zeros_like(mjds)
ss_roemer = numpy.zeros_like(mjds)
ss_shapiro = numpy.zeros_like(mjds)

sys.stderr.write("Computing residuals...\n")
t0 = time.time()
for ii, tt in enumerate(t.toas):
    p = m.phase(tt)
    resids[ii] = p.frac
    ss_roemer[ii] = m.solar_system_geometric_delay(tt)
    ss_shapiro[ii] = m.solar_system_shapiro_delay(tt)

time_phase = time.time() - t0
sys.stderr.write("Computed phases in %.3f sec\n" % time_phase)

# resids in (approximate) us:
resids_us = resids / float(m.F0.value) * 1e6
sys.stderr.write("RMS PINT residuals are %.3f us\n" % resids_us.std())

# Get some general2 stuff
tempo2_vals = tempo2_utils.general2(parfile, timfile,
                                    ['tt2tb', 'roemer', 'post_phase',
                                     'shapiro', 'shapiroJ'])
t2_resids = tempo2_vals['post_phase'] / float(m.F0.value) * 1e6
diff_t2 = resids_us - t2_resids
diff_t2 -= diff_t2.mean()

# run tempo1 also, if the tempo_utils module is available
try:
    import tempo_utils
    t1_toas = tempo_utils.read_toa_file(timfile)
    tempo_utils.run_tempo(t1_toas, t1_parfile)
    t1_resids = t1_toas.get_resids(units='phase') / float(m.F0.value) * 1e6
    diff_t1 = resids_us - t1_resids
    diff_t1 -= diff_t1.mean()

    diff_t2_t1 = t2_resids - t1_resids
    diff_t2_t1 -= diff_t2_t1.mean()
except:
    pass

def do_plot():
    plt.clf()
    plt.subplot(211)
    plt.hold(False)
    plt.errorbar(mjds,resids_us,errs,fmt=None,label='PINT')
    plt.title("J1744-1134 GBT/GASP timing")
    plt.xlabel('MJD')
    plt.ylabel('Residual (us)')
    plt.legend()
    plt.grid()
    plt.subplot(212)
    plt.plot(mjds,diff_t2*1e3,label='PINT - T2')
    plt.hold(True)
#    plt.plot(mjds,diff_t1*1e3,label='PINT - T1')
    plt.grid()
    plt.xlabel('MJD')
    plt.ylabel('Residual diff (ns)')
    plt.legend()

#do_plot()
#plt.show()
