import matplotlib.pyplot as plt
import mpmath as mp
from spiceTest import *
from astropy.time import Time
import HPtimeCvrtLib as tc
import toa as toas

mp.mp.dps = 25 

mjdJ2000 = mp.mpf("51544.5")
secDay = mp.mpf("86400.0")


#### From mjdutc to et
def mjd2et(mjd,tt2tdb):
    """
    Convert from mjd utc to et using tempo2 out put tt2tdb
    """
    mjdJ2000 = mp.mpf("51544.5")
    secDay = mp.mpf("86400.0")
    mjdTT = tc.mjd2tdt(mp.mpf(mjd)) 
    # Convert mjdutc to mjdtdt using HP time convert lib 
   # print "python ",mjdTT
    et = (mp.mpf(mjdTT)-mjdJ2000)*mp.mpf(86400.0)+mp.mpf(tt2tdb)
    return et
#### Read tempo2 tim file
fname1 = "J0000+0000.tim"
fp1 = open(fname1,"r")
toa = [] 
# Read TOA column to toa array
for l in fp1.readlines():
    l = l.strip()
    l = l.strip("\n")
    l = l.split()
    if(len(l)>3):
        toa.append(l[2])

#### Read tempo2 general2 output file
fname ="T2output.dat"
fp = open(fname,"r")

tt2tdb = []    # Tempo2 tt2tdb difference in (sec)
earth1 =[]     # Tempo2 earth position in (light time, sec)
earth2 =[]
earth3 =[]
# Read tt2tdb earthposition output
for l in fp.readlines():
	l = l.strip()
	l = l.strip("\n")
	l = l.split()
# Avoid the column that is not data
	try:                
		m = float(l[0]) 
	except:
		pass
	else:
		tt2tdb.append(l[-1]) 
		earth1.append(l[0])
		earth2.append(l[1])
		earth3.append(l[2])
#### Use toa structure. 
t1 = toas.TOAs(fname1)
#### Testing toa mjd to tt
tt = []
ttToa = []
for i in range(len(toa)):
    tt.append(tc.mjd2tdt(mp.mpf(toa[i])))
    ttToa.append(mp.mpf(t1.toas[i].mjd.tt.jd1)+mp.mpf(t1.toas[i].mjd.tt.jd2)\
    -mp.mpf(2400000.5))
print ttToa[0]

##### Testing the delta_tdb_tt from astropy 
lon = -66.7527926727 
lat = 18.2296075635
Dtttdb = []
for i in range(len(toa)):
    ii, ff = toa[i].split('.')
    MJD = (int(ii), float("0."+ff))
    tAstropy = Time(*MJD,format = 'mjd',scale = 'utc',lat=lat, lon=lon, 
    precision=9)
#    print mp.mpf(tAstropy.utc.jd1)+mp.mpf(tAstropy.utc.jd2),data
    Dtttdb.append(tAstropy.tt.delta_tdb_tt)
    print tAstropy.tt.delta_tdb_tt,tt2tdb[i]   
# Testing Convert toa mjd to toa et        
et = []
etToa = []
for i in range(len(toa)):
#    toa[i] = float(toa[i])
    et.append(mjd2et(toa[i],tt2tdb[i]))
    etTemp = (ttToa[i]-mjdJ2000)*mp.mpf(86400.0)+mp.mpf(tt2tdb[i])
    etToa.append(etTemp)
print et[0],etToa[0]    
###### calculate earth position
stateInterp = []     # interpolated earth position   in (km)    
ltInterp = []        # interpolated earth to ssb light time in (sec)
statespk = []        # Directlt calculated earth position in (km)
ltspk = []           # Directly calculated earth to ssb lt time in (sec)
# Calculating postion
for time in etToa:
    state0,lt0, = spkInterp(float(time),7)
    stateInterp.append(state0)
    ltInterp.append(lt0)
    state1,lt1 = spice.spkezr("EARTH",time,"J2000","NONE","SSB")
    statespk.append(list(state1))
    ltspk.append(lt1)
# Print the result
print stateInterp[0]   # Interpolated earth position
print statespk[0]      # Directly calculate
# Tempo2 earth position in (km)
print mp.mpf(earth1[0])*mp.mpf(spice.clight()),\
    mp.mpf(earth2[0])*mp.mpf(spice.clight()),\
    mp.mpf(earth3[0])*mp.mpf(spice.clight())

# Plot
stateInterpy = []
statespky = []
diff1 = []
diff2=[]
diffet = []
for i in range(len(statespk)):
    stateInterpy.append(mp.mpf(stateInterp[i][1]))
    statespky.append(mp.mpf(statespk[i][1]))
    diff1.append(stateInterpy[i]-statespky[i])
# Difference between interploated position and tempo2 postion out put in KM
    diff2.append(stateInterpy[i]-mp.mpf(earth2[i])*mp.mpf(spice.clight()))

plt.figure(1)
plt.title("T2 output and interpolated spice out put difference in (km) using toa class")
plt.ylabel("y axis Position difference in km")
plt.plot(diff2)
plt.show()
