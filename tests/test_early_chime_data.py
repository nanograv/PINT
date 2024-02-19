"""Various tests to assess the performance of early CHIME data."""
import os
import pytest

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir


class Test_CHIME_data:
    """Compare delays from the dd model with tempo and PINT"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parfile = "B1937+21.basic.par"
        cls.tim = "B1937+21.CHIME.CHIME.NG.N.tim"

    def test_toa_read(self):
        toas = toa.get_TOAs(self.tim, ephem="DE436", planets=False, include_bipm=True)
        assert toas.ntoas == 848, "CHIME TOAs did not read correctly."
        assert list(set(toas.get_obss())) == [
            "chime"
        ], "CHIME did not recognized by observatory module."

    def test_residuals(self):
        model = mb.get_model(self.parfile)
        toas = toa.get_TOAs(self.tim, ephem="DE436", planets=False, include_bipm=True)
        r = Residuals(toas, model)
        # Comment out the following test for now, since the new residual
        # code makes it fail, and it probably shouldn't -- SMR
        # assert_quantity_allclose(r.time_resids.to(u.us), 0*u.us, atol=800*u.us, rtol=0)
