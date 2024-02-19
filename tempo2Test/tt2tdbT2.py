from __future__ import print_function, division
import mpmath as mp
from spiceTest import *

mp.mp.dps = 25


#### From mjd2tdt
def mjd2tdt(mjd):
    """
    Convert from mjd utc to mjd tdt using mpmath
    """
    dt = 10.0
    if mjd >= 41499.0:
        dt = 11.0  # /* 1972 Jul 1 */
    if mjd >= 41683.0:
        dt = 12.0
        # /* 1973 Jan 1 */
    if mjd >= 42048.0:
        dt = 13.0
        # /* 1974 Jan 1 */
    if mjd >= 42413.0:
        dt = 14.0
        # /* 1975 Jan 1 */
    if mjd >= 42778.0:
        dt = 15.0
        # /* 1976 Jan 1 */
    if mjd >= 43144.0:
        dt = 16.0
        # /* 1977 Jan 1 */
    if mjd >= 43509.0:
        dt = 17.0
        # /* 1978 Jan 1 */
    if mjd >= 43874.0:
        dt = 18.0
        # /* 1979 Jan 1 */
    if mjd >= 44239.0:
        dt = 19.0
        # /* 1980 Jan 1 */
    if mjd >= 44786.0:
        dt = 20.0
        # /* 1981 Jul 1 */
    if mjd >= 45151.0:
        dt = 21.0
        # /* 1982 Jul 1 */
    if mjd >= 45516.0:
        dt = 22.0
        # /* 1983 Jul 1 */
    if mjd >= 46247.0:
        dt = 23.0
        # /* 1985 Jul 1 */
    if mjd >= 47161.0:
        dt = 24.0
        # /* 1988 Jan 1 */
    if mjd >= 47892.0:
        dt = 25.0
        # /* 1990 Jan 1 */
    if mjd >= 48257.0:
        dt = 26.0
        # /* 1991 Jan 1 */
    if mjd >= 48804.0:
        dt = 27.0
        # /* 1992 July 1 */
    if mjd >= 49169.0:
        dt = 28.0
        # /* 1993 July 1 */
    if mjd >= 49534.0:
        dt = 29.0
        # /* 1994 July 1 */
    if mjd >= 50083.0:
        dt = 30.0
        # /* 1996 Jan 1 */
    if mjd >= 50630.0:
        dt = 31.0
        # /* 1997 Jul 1 */
    if mjd >= 51179.0:
        dt = 32.0
        # /* 1999 Jan 1 */
    if mjd >= 53736.0:
        dt = 33.0
        # /* 2006 Jan 1 */
    if mjd >= 54832.0:
        dt = 34.0
        # /* 2009 Jan 1 */
    if mjd >= 56109.0:
        dt = 35.0
        # /* 2012 July 1 */

    delta_TT = mp.mpf(dt) + 32.184
    delta_TT_DAY = mp.mpf(delta_TT) / mp.mpf(86400.0)
    delta_TT_DAY = mp.mpf(delta_TT_DAY)
    return mp.mpf(mjd) + delta_TT_DAY


#### From mjdutc to et
def mjd2et(mjd, tt2tdb):
    """
    Convert from mjd utc to et using tempo2 out put tt2tdb
    """
    mjdJ2000 = mp.mpf("51544.5")
    secDay = mp.mpf("86400.0")
    mjdTT = mjd2tdt(mjd)
    return (mjdTT - mjdJ2000) * secDay + mp.mpf(tt2tdb)


#### Read tempo2 tim file
fname1 = "J0000+0000.tim"
fp1 = open(fname1, "r")

toa = []
# Read TOA column to toa array
for l in fp1:
    l = l.strip()
    l = l.strip("\n")
    l = l.split()
    if len(l) > 3:
        toa.append(l[2])

#### Read tempo2 general2 output file
fname = "T2output.dat"
fp = open(fname, "r")

tt2tdb = []  # Tempo2 tt2tdb difference in (sec)
earth1 = []  # Tempo2 earth position in (light time, sec)
earth2 = []
earth3 = []
# Read tt2tdb earth position output
for l in fp:
    l = l.strip()
    l = l.strip("\n")
    l = l.split()
    # Avoid the column that is not data
    try:
        m = float(l[0])
    except Exception:
        pass
    else:
        tt2tdb.append(l[-1])
        earth1.append(l[0])
        earth2.append(l[1])
        earth3.append(l[2])

et = [mjd2et(toa[i], tt2tdb[i]) for i in range(len(toa))]
###### calculate earth position
stateInterp = []  # interpolated earth position   in (km)
ltInterp = []  # interpolated earth to ssb light time in (sec)
statespk = []  # Directly calculated earth position in (km)
ltspk = []  # Directly calculated earth to ssb lt time in (sec)
# Calculating position
for time in et:
    (
        state0,
        lt0,
    ) = spkInterp(float(time), 4)
    stateInterp.append(state0)
    ltInterp.append(lt0)
    state1, lt1 = spice.spkezr("EARTH", time, "J2000", "NONE", "SSB")
    statespk.append(state1)
    ltspk.append(lt1)
# Print the result
print(stateInterp[0])  # Interpolated earth position
print(statespk[0])  # Directly calculate
# Tempo2 earth position in (km)
print(
    mp.mpf(earth1[0]) * mp.mpf(spice.clight()),
    mp.mpf(earth2[0]) * mp.mpf(spice.clight()),
    mp.mpf(earth3[0]) * mp.mpf(spice.clight()),
)
