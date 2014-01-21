import mpmath as mp

mp.mp.dps = 25
spd = 86400.0    # seconds per day
hpd = 24.0       # hours per day

initT = 49718.5  # initial time in MJD (J2000 starting time)

timeArray = []

NumData = 20*365*24

f = open("20yrsTOA.tim","w")
outPut = 'FORMAT 1\n'
f.write(outPut)
for i in range(NumData):
    timeArray.append(mp.mpf(initT) + i*mp.mpf(1.0)/mp.mpf(hpd))
    outPut =  'c01_J0000+0000'+' 1440.0 '+mp.nstr(timeArray[i],22)+\
    ' .0000001'+' ao'+'\n'
    f.write(outPut)



