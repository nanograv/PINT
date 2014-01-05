import os
import numpy
import spice
def read_observatories():
    obss = {}
    obscode1s = {}
    obscode2s = {}
    filenm = os.path.join(os.getenv("PINT"), "datafiles/observatories.txt")
    with open(filenm) as f:
        for line in f.readlines():
            if line[0]!="#":
                name, x, y, z, code1, code2 = line.split()
                obss[name.upper()] = dict(xyz=(float(x), float(y), float(z)))
                obscode1s[code1] = name.upper()
                obscode2s[code2] = name.upper()
    return obss, obscode1s, obscode2s

#def ITRF2Geo()



