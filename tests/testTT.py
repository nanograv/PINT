#! /usr/bin/env python
from pint import toa
import libstempo as libt
import tempo2_utils as t2u
import mpmath as mp
mp.mp.dps = 25
spd = mp.mpf(86400.0)


tt0 = []
tt1 = []
t = toa.TOAs('J1744-1134.Rcvr1_2.GASP.8y.x.tim')
for i in range(len([x for x in t.toas])):
    tt0.append(mp.mpf(t.toas[i].mjd.tt.jd1)+mp.mpf(t.toas[i].mjd.tt.jd2))
t.apply_clock_corrections()

for i in range(len([x for x in t.toas])):
     tt1.append(mp.mpf(t.toas[i].mjd.tt.jd1)+mp.mpf(t.toas[i].mjd.tt.jd2))
diff = []
for i in range(len(tt0)): 
     diff.append((tt0[i]-tt1[i])*spd)


ttT2 = t2u.general2('J1744-1134.basic.par','J1744-1134.Rcvr1_2.GASP.8y.x.tim',['tt'])
 
print ttT2 
 
     
