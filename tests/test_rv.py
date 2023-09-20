import pytest
import numpy as np
from astropy import units as u, constants as c

from pint.models import get_model
import pint.simulation
import pint.binaryconvert
from pinttestdata import datadir
import os


class TestRV:
    def setup_method(self):
        self.m = get_model(
            os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_modified_DD.par")
        )
        # set the eccentricity to nonzero
        # but not too high so that the models with various approximations won't fail
        self.m.ECC.value = 0.01
        self.ts = self.m.T0.value + np.linspace(0, 2 * self.m.PB.value, 101)
        self.ts = pint.simulation.make_fake_toas_fromMJDs(self.ts, self.m)

        self.v = self.m.pulsar_radial_velocity(self.ts)

    def test_rv_basemodel(self):
        nu = self.m.orbital_phase(self.ts, anom="true", radians=True)
        # HBOPA 8.24
        v = (
            (2 * np.pi / self.m.PB.quantity)
            * (self.m.A1.quantity / np.sqrt(1 - self.m.ECC.quantity**2))
            * (
                np.cos(nu + self.m.OM.quantity)
                + self.m.ECC.quantity * np.cos(self.m.OM.quantity)
            )
        )
        assert np.allclose(self.v, v)

    @pytest.mark.parametrize("othermodel", ["ELL1", "ELL1H", "DDS", "BT"])
    def test_rv_othermodels(self, othermodel):
        mc = pint.binaryconvert.convert_binary(self.m, othermodel)
        vc = mc.pulsar_radial_velocity(self.ts)
        # have a generous tolerance here since some of the models don't work well for high ECC
        assert np.allclose(vc, self.v, atol=20 * u.km / u.s, rtol=1e-2)
