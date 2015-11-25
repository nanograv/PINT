from __future__ import print_function, division
import matplotlib.pyplot as plt
import libstempo as lt
import pint.toa as toa
import pint.models.bt as bt

def produce_libstempo_delays():
    """Use simulated data of J1955 with TEMPO2 and test if reproducible"""

    parfile = 'tests/J1955.par'
    timfile = 'tests/J1955.tim'

    toas = toa.get_TOAs(timfile, planets=False, usepickle=False)

    SECS_PER_DAY = 86400

    toasld = toas.table['tdbld']
    t0 = toasld[0]
    toaslds = (toasld - t0) * SECS_PER_DAY

    newmodel = bt.BT()

    newmodel.read_parfile(parfile)

    phases = newmodel.phase(toas.table)

    #t0 = time.time()
    delays = newmodel.delay(toas.table)

    psr = lt.tempopulsar(parfile, timfile)
    t2resids = psr.residuals()
    btdelay = psr.binarydelay()
    bttoasld = psr.toas()
    btt0 = bttoasld[0]
    bttoaslds = (bttoasld - btt0) * SECS_PER_DAY

    # HACK (paulr): Flip sign of btdelay to match what PINT uses.  Not sure why this is necessary. Should be figured out.
    btdelay *= -1.0

    with open('tests/J1955_ltdelays.dat', 'w') as fobj:
        for i in range(len(delays)):
            print("{:.20} {:.20}".format(toaslds[i], btdelay[i]), file=fobj)

    plt.figure("Delays")
    plt.plot(toaslds, delays, label='PINT')
    plt.scatter(toaslds, delays)

    plt.plot(bttoaslds, btdelay, label='T2')
    plt.scatter(bttoaslds, btdelay)

    plt.legend()
    plt.xlabel('Time (d)')
    plt.ylabel('Delay (s)')

    plt.figure("Diff")

    plt.plot(toaslds, delays - btdelay)
    plt.scatter(toaslds, delays - btdelay)

    plt.legend()
    plt.xlabel('Time (d)')
    plt.ylabel('Diff (s)')
    plt.show()

if __name__ == '__main__':
    produce_libstempo_delays()
