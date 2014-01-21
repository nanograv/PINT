import sys
from pint import toa, utils
import tempo2_utils as t2u
#t = toa.TOAs('20yrsTOA.tim')     
d_tt_tdb_SPICE = []

def getTTmTDB(parFile,timFile):
    result = t2u.general2(parFile,timFile,['tt2tb'])
    t = toa.TOAs(timFile)
    d_tt_tdb_FB90 = []
    d_tt_tdb_IF99 = []
    for i in range(len([x for x in t.toas])):
        d_tt_tdb_FB90.append(t.toas[i].mjd.tt.delta_tdb_tt[0])
        d_tt_tdb_IF99.append(result['tt2tb'][i])

        f = open("tt2tdb.dat",'w')
    for l in range(len(d_tt_tdb_FB90)):
        outPut = str(d_tt_tdb_FB90[l])+' '+str(d_tt_tdb_IF99[l])+'\n'
        f.write(outPut)
    f.close()    
def getTDT(parFile,timFile):
    result = t2u.general2(parFile,timFile,['tt'])
    t = toa.TOAs(timFile)
    tt_tempo = []
    tt_astropy = []
    for i in range(len([x for x in t.toas])):
        tt_tempo.append(result['tt'][i])
        tt_astropy.append([t.toas[i].mjd.tt.jd1,t.toas[i].mjd.tt.jd2])
    print tt_tempo,tt_astropy        
