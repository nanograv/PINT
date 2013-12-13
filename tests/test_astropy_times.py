import astropy.time

lat = 19.48125
lon = -155.933222
t = astropy.time.Time('2006-01-15 21:24:37.5', \
                      format='iso', scale='utc', \
                      lat=lat, lon=lon, precision=6)
print "Checking UTC"
x = t.utc.iso
y = '2006-01-15 21:24:37.500000'
assert(x==y)
print "Checking UTC-UT1"
x = t.ut1.iso
y = '2006-01-15 21:24:37.834078'
assert(x==y)
print "Checking UT1-TAI"
x = t.tai.iso
y = '2006-01-15 21:25:10.500000'
assert(x==y)
print "Checking TAI-TT"
x = t.tt.iso
y = '2006-01-15 21:25:42.684000'
assert(x==y)
print "Checking TT-TCG"
x = t.tcg.iso
y = '2006-01-15 21:25:43.322690'
assert(x==y)
print "Checking TT-TDB"
x = t.tdb.iso
y = '2006-01-15 21:25:42.683799'
assert(x==y)
print "Checking TT-TCB"
x = t.tcb.iso
y = '2006-01-15 21:25:56.893378'
assert(x==y)


import astropy.utils
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
iers_a_file = download_file(IERS_A_URL, cache=True)
iers_a = IERS_A.open(iers_a_file)
t2 = astropy.time.Time.now()
t2.delta_ut1_utc = t2.get_delta_ut1_utc(iers_a)
print t2.tdb.iso

# From the SOFA manual:
"""
UTC 2006/01/15 21:24:37.500000
UT1 2006/01/15 21:24:37.834100
TAI 2006/01/15 21:25:10.500000
TT 2006/01/15 21:25:42.684000
TCG 2006/01/15 21:25:43.322690
TDB 2006/01/15 21:25:42.683799
TCB 2006/01/15 21:25:56.893378
"""
