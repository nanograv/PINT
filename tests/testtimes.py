import toa, utils
import math, shlex, subprocess
import astropy.units as u

ts = toa.TOAs("tests/testtimes.tim")
# print ts.toas[0].mjd.lat, ts.toas[0].mjd.lon
ts.apply_clock_corrections()
# print ts.toas[0].mjd.lat, ts.toas[0].mjd.lon
# print ts.summary()

cmd = 'tempo2 -output general2 -f tests/testtimes.par tests/testtimes.tim -s "XXX {clock0} {clock1} {clock2} {clock3} {tt} {t2tb} {Ttt}\n"'
args = shlex.split(cmd)

tout = subprocess.check_output(args)
goodlines = [x for x in tout.split("\n") if x.startswith("XXX")]

assert(len(goodlines)==len(ts.toas))

# Get the output lines from the TOAs
for line, toa in zip(goodlines, ts.toas):
    oclk, ut1_utc, tai_utc, tt_tai, ttcorr, tt2tb, Ttt = \
          (float(x) for x in line.split()[1:])
    # print utils.time_to_mjd_string(toa.mjd.tt), line.split()[-1]
    tempo_tt = utils.time_from_mjd_string(line.split()[-1], scale='tt') 
    # Ensure that the clock corrections are accurate to better than 0.1 ns
    assert(math.fabs((oclk*u.s - toa.flags["clkcorr"]).to(u.ns).value) < 0.1)

    # Where in the heck is TEMPO2's ut1-utc correction?!?
    print toa.mjd.delta_ut1_utc[0], ut1_utc, ttcorr-tai_utc-tt_tai, \
          toa.mjd.tdb.delta_tdb_tt[0], tt2tb-ttcorr
    print "TOA in tt difference is:", \
          ((toa.mjd.tt - tempo_tt.tt).sec * u.s).to(u.ns)
