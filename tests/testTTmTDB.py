import mpmath as mp

spd = 86400.0    # seconds per day
hpd = 24.0       # hours per day

initT = 51544.5  # initial time in MJD (J2000 starting time)

timeArray = []

NumData = 20*365*24


for i in range(NumData):
    timeArray.append(initT + i*1.0/hpd)
    print timeArray[i]

