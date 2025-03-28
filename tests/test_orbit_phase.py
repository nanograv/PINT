import pytest
import os
import pytest

import numpy as np

import pint.toa as t
import pint.models as m
from pinttestdata import datadir


class TestOrbitPhase:
    """Test orbital phase calculations"""

    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.pJ1855 = "B1855+09_NANOGrav_dfg+12_modified_DD.par"
        cls.mJ1855 = m.get_model(cls.pJ1855)

    def test_barytimes(self):
        ts = t.Time([56789.234, 56790.765], format="mjd")
        # Should raise ValueError since not in "tdb"
        with pytest.raises(ValueError):
            self.mJ1855.orbital_phase(ts)
        # Should raise ValueError since not correct anom
        with pytest.raises(ValueError):
            self.mJ1855.orbital_phase(ts.tdb, anom="xxx")
        # Should return
        phs = self.mJ1855.orbital_phase(ts.tdb, anom="mean")
        assert len(phs) == 2
        toas = t.get_TOAs("test1.tim")
        phs = self.mJ1855.orbital_phase(toas)
        assert len(phs) == toas.ntoas

    def test_j1855_nonzero_ecc(self):
        ts = self.mJ1855.T0.value + np.linspace(0, self.mJ1855.PB.value, 101)
        self.mJ1855.ECC.value = 0.1  # set the eccentricity to nonzero
        phs = self.mJ1855.orbital_phase(ts, anom="mean", radians=False)
        assert np.all(phs >= 0), "Not all phases >= 0"
        assert np.all(phs <= 1), "Not all phases <= 1"
        phs = self.mJ1855.orbital_phase(ts, anom="mean")
        assert np.all(phs.value >= 0), "Not all phases >= 0"
        assert np.all(phs.value <= 2 * np.pi), "Not all phases <= 2*pi"
        phs2 = self.mJ1855.orbital_phase([ts[0], ts[49]], anom="ecc")
        assert np.isclose(phs2[0].value, phs[0].value), "Eccen anom != Mean anom"
        assert phs2[1] != phs[49], "Eccen anom == Mean anom"
        phs3 = self.mJ1855.orbital_phase([ts[0], ts[49]], anom="true")
        assert np.isclose(phs3[0].value, phs[0].value), "Eccen anom != True anom"
        assert phs3[1] != phs[49], "Eccen anom == True anom"

    def test_j1855_zero_ecc(self):
        self.mJ1855.ECC.value = 0.0  # set the eccentricity to zero
        self.mJ1855.OM.value = 0.0  # set omega to zero
        phs1 = self.mJ1855.orbital_phase(self.mJ1855.T0.value, anom="mean")
        assert np.isclose(phs1.value, 0.0), "Mean anom != 0.0 at T0"
        # All anomalies are equivalent when ECC = OM = 0
        phs1 = self.mJ1855.orbital_phase(self.mJ1855.T0.value + 0.1, anom="mean")
        phs2 = self.mJ1855.orbital_phase(self.mJ1855.T0.value + 0.1, anom="ecc")
        assert np.isclose(phs2.value, phs1.value), "Eccen anom != Mean anom"
        phs3 = self.mJ1855.orbital_phase(self.mJ1855.T0.value + 0.1, anom="true")
        assert np.isclose(phs3.value, phs1.value), "True anom != Mean anom"

    def test_j1855_ell1(self):
        mJ1855ell1 = m.get_model("B1855+09_NANOGrav_12yv3.wb.gls.par")
        phs1 = mJ1855ell1.orbital_phase(mJ1855ell1.TASC.value, anom="mean")
        assert np.isclose(phs1.value, 0.0), "Mean anom != 0.0 at TASC as value"
        phs1 = mJ1855ell1.orbital_phase(mJ1855ell1.TASC, anom="mean")
        assert np.isclose(phs1.value, 0.0), "Mean anom != 0.0 at TASC as MJDParam"
        # All anomalies are equivalent in ELL1
        phs1 = mJ1855ell1.orbital_phase(mJ1855ell1.TASC.value + 0.1, anom="mean")
        phs2 = mJ1855ell1.orbital_phase(mJ1855ell1.TASC.value + 0.1, anom="ecc")
        assert np.isclose(phs2.value, phs1.value), "Eccen anom != Mean anom"
        phs3 = mJ1855ell1.orbital_phase(mJ1855ell1.TASC.value + 0.1, anom="true")
        assert np.isclose(phs3.value, phs1.value), "True anom != Mean anom"

    def test_j0737(self):
        # Find a conjunction which we have confirmed by GBT data and Shapiro delay
        mJ0737 = m.get_model("0737A_latest.par")
        x = mJ0737.conjunction(55586.25)
        assert np.isclose(x, 55586.29643451057), "J0737 conjunction time is bad"
        # And now make sure we calculate the true anomaly for it correctly
        nu = mJ0737.orbital_phase(x, anom="true").value
        omega = mJ0737.components["BinaryDD"].binary_instance.omega().value
        # Conjunction occurs when nu + OM == 90 deg
        assert np.isclose(
            np.degrees(np.remainder(nu + omega, 2 * np.pi)), 90.0
        ), "J0737 conjunction time gives bad true anomaly"
        # Now verify we can get 2 results from .conjunction
        x = mJ0737.conjunction([55586.0, 55586.2])
        assert len(x) == 2, "conjunction is not returning an array"
        # make sure true anomaly before T0 is positive
        assert mJ0737.orbital_phase(52000.0, anom="true").value > 0.0
