import libstempo as T
import numpy as np
import datetime

timJ1713 = "J1713+0747_NANOGrav_11yv0_short.tim"
# one file in ICRS, one in ECL
parfiles = [
    "J1713+0747_NANOGrav_11yv0_short.gls.ICRS.par",
    "J1713+0747_NANOGrav_11yv0_short.gls.par",
]
for parfile in parfiles:
    psr = T.tempopulsar(parfile=parfile, timfile=timJ1713)
    x = np.array([psr.residuals(), psr.binarydelay()])
    fout = open(parfile + ".libstempo", "w")
    fout.write(
        "# Created using libstempo on %s\n" % datetime.datetime.now().isoformat()
    )
    fout.write("# Residual BinaryDelay\n")
    for i in range(psr.nobs):
        fout.write("%e %e\n" % (x[0, i], x[1, i]))
    fout.close()
