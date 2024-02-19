import sys

import astropy.time as time
import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa


def add_FD_model(freq_range, FD, toas):
    """This function is to add a FD model delay in the toas, the frequency will
    be changed as well
    Parameter
    ----------
    freq_range : ndarray
        The range of frequency, [start_frequency, end_frequency] in MHz
    FD : ndarray
        FD model coefficents [FDn, FDn-1, ..., FD2, FD1]
    toas : PINT toa class
        The toas need to be changed.
    """
    ntoas = toas.ntoas
    freq = np.linspace(freq_range[0], freq_range[1], ntoas) * u.MHz
    fdcoeff = FD + [0.0]
    logfreq = np.log(freq / (1 * u.GHz))
    fd_delay = np.polyval(fdcoeff, logfreq) * u.second
    dt = time.TimeDelta(fd_delay)
    toas.adjust_TOAs(dt)
    toas.table["freq"] = freq
    return toas


if __name__ == "__main__":
    timfile = sys.argv[1]
    parfile = sys.argv[2]
    freq_start = float(sys.argv[3])
    freq_end = float(sys.argv[4])

    t = toa.get_TOAs(timfile)
    m = mb.get_model(parfile)
    freqrange = np.array([freq_start, freq_end])
    fdcoeff = [
        getattr(m, m.FDmapping[ii]).num_value for ii in range(m.num_FD_terms, 0, -1)
    ]

    add_FD_model(freqrange, fdcoeff, t)
    outfile = f"{timfile}.pint_simulate"
    t.write_TOA_file(outfile, format="TEMPO2")
