from pint import toa, utils
import math, shlex, subprocess, numpy
import astropy.constants as const
import astropy.units as u

ls = u.def_unit('ls', const.c * 1.0 * u.s)

ts = toa.TOAs("tests/testtimes.tim")
# print ts.toas[0].mjd.lat, ts.toas[0].mjd.lon
ts.apply_clock_corrections()
ts.compute_posvels()
# print ts.toas[0].mjd.lat, ts.toas[0].mjd.lon
# print ts.summary()

#cmd = 'tempo2 -output general2 -f tests/testtimes.par tests/testtimes.tim -s "XXX {clock0} {clock1} {clock2} {clock3} {tt} {t2tb} {telSSB} {telVel} {Ttt}\n"'
cmd = 'tempo2 -output general2 -f tests/testtimes.par tests/testtimes.tim -s "XXX {clock0} {clock1} {clock2} {clock3} {tt} {t2tb} {earth_ssb1} {earth_ssb2} {earth_ssb3} {earth_ssb4} {earth_ssb5} {earth_ssb6} {telEpos} {telEVel} {Ttt}\n"'
args = shlex.split(cmd)

tout = subprocess.check_output(args)
goodlines = [x for x in tout.split("\n") if x.startswith("XXX")]

assert(len(goodlines)==len(ts.toas))

# Get the output lines from the TOAs
for line, toa in zip(goodlines, ts.toas):
    oclk, ut1_utc, tai_utc, tt_tai, ttcorr, tt2tb, \
          ep0, ep1, ep2, ev0, ev1, ev2, \
          tp0, tp1, tp2, tv0, tv1, tv2, Ttt = \
          (float(x) for x in line.split()[1:])
    t2_epv = utils.PosVel(numpy.asarray([ep0, ep1, ep2]) * ls,
                          numpy.asarray([ev0, ev1, ev2]) * ls/u.s)
    t2_opv = utils.PosVel(numpy.asarray([tp0, tp1, tp2]) * ls,
                          numpy.asarray([tv0, tv1, tv2]) * ls/u.s)
    # print utils.time_toq_mjd_string(toa.mjd.tt), line.split()[-1]
    tempo_tt = utils.time_from_mjd_string(line.split()[-1], scale='tt') 
    # Ensure that the clock corrections are accurate to better than 0.1 ns
    assert(math.fabs((oclk*u.s - toa.flags["clkcorr"]).to(u.ns).value) < 0.1)

    print "\nTOA in tt difference is:", \
          ((toa.mjd.tt - tempo_tt.tt).sec * u.s).to(u.ns)

    print "SSB-Earth:"
    print "   T2:", t2_epv.pos.to(ls), t2_epv.vel.to(ls/u.s)
    print " PINT:", toa.earth_pvs.pos.to(ls), toa.earth_pvs.vel.to(ls/u.s)
    dssb = toa.earth_pvs - t2_epv
    print " diff:", dssb.pos.to(ls), dssb.vel.to(ls/u.s)

    print "topocenter:"
    print "   T2:", t2_opv.pos.to(u.m), t2_opv.vel.to(u.m/u.s)
    print " PINT:", toa.obs_pvs.pos.to(u.m), toa.obs_pvs.vel.to(u.m/u.s)
    dgeo = toa.obs_pvs - t2_opv
    xx = dgeo.pos / numpy.sqrt(numpy.dot(toa.obs_pvs.pos, toa.obs_pvs.pos))
    xx = numpy.sqrt(numpy.dot(xx, xx))
    print (xx * u.rad).to(u.arcsec)
    print " diff:", dgeo.pos.to(u.m), dgeo.vel.to(u.m/u.s)

    # Where in the heck is TEMPO2's ut1-utc correction?!?
    #print toa.mjd.delta_ut1_utc[0], ut1_utc, ttcorr-tai_utc-tt_tai, \
    #      toa.mjd.tdb.delta_tdb_tt[0], tt2tb-ttcorr
