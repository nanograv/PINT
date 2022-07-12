"""tests for different compute TDB method."""
import os

import numpy as np
from pinttestdata import datadir

import pint.toa as toa


def test_astropy_ephem(pickle_dir):
    tim = datadir / "B1855+09_NANOGrav_9yv1.tim"
    t_astropy = toa.get_TOAs(tim, ephem="DE436t", picklefilename=pickle_dir)
    t_ephem = toa.get_TOAs(
        tim, ephem="DE436t", tdb_method="ephemeris", picklefilename=pickle_dir
    )
    diff = (t_astropy.table["tdbld"] - t_ephem.table["tdbld"]) * 86400.0
    assert np.all(np.abs(diff) < 5e-9)
