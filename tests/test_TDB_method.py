"""tests for different compute TDB method."""
import os
import pytest

import numpy as np

import pint.toa as toa
from pinttestdata import datadir


class TestTDBMethod:
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.tim = "B1855+09_NANOGrav_9yv1.tim"

    def test_astropy_ephem(self):
        t_astropy = toa.get_TOAs(self.tim, ephem="DE436t")
        t_ephem = toa.get_TOAs(self.tim, ephem="DE436t", tdb_method="ephemeris")
        diff = (t_astropy.table["tdbld"] - t_ephem.table["tdbld"]) * 86400.0
        assert np.all(np.abs(diff) < 5e-9), (
            "Test TDB method, 'astropy' vs " "'ephemeris' failed."
        )
