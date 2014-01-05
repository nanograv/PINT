import HPtimeCvrtLib as tc
import mpmath as mp

mp.mp.dps = 30

filename = "T2output.dat"
fp = open(filename,"r")




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
        
# Calculate et
et = [] 
for mjd in toa:
    et.append(tc.jdutc2et(mp.mpf(mjd)+mp.mpf(2400000.5)))
