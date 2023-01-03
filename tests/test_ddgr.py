from astropy import units as u, constants as c
import numpy as np
import io
import os

from pint.models import get_model
from pint import derived_quantities
import pint.simulation

par = """
PSRJ           1855+09
RAJ             18:57:36.3932884         0  0.00002602730280675029
DECJ           +09:43:17.29196           0  0.00078789485676919773
F0             186.49408156698235146     0  0.00000000000698911818
F1             -6.2049547277487420583e-16 0  1.7380934373573401505e-20
PEPOCH         49453
POSEPOCH       49453
DMEPOCH        49453
DM             13.29709
PMRA           -2.5054345161030380639    0  0.03104958261053317181
PMDEC          -5.4974558631993817232    0  0.06348008663748286318
PX             1.2288569063263405232     0  0.21243361289239687251
T0             49452.940695077335647     0  0.00169031830532837251
OM             276.55142180589701234     0  0.04936551005019605698
ECC            0.1 0  0.00000004027191312623
START          53358.726464889485214
FINISH         55108.922917417192366
TZRMJD         54177.508359343262555
TZRFRQ         424
TZRSITE        ao
TRES           0.395
EPHVER         5
CLK            TT(TAI)
MODE 1
UNITS          TDB
T2CMETHOD      TEMPO
#NE_SW          0.000
CORRECT_TROPOSPHERE  N
EPHEM          DE405
NITS           1
NTOA           702
CHI2R          2.1896 637
SOLARN0        00.00
TIMEEPH        FB90
PLANET_SHAPIRO N
"""
Mp = 1.4 * u.Msun
Mc = 1.1 * u.Msun
i = 85 * u.deg
PB = 0.5 * u.day


class TestDDGR:
    def setup(self):
        A1 = derived_quantities.a1sini(Mp, Mc, PB, i)
        self.mDD = get_model(
            io.StringIO(
                f"{par}\nBINARY DD\nSINI {np.sin(i).value}\nA1 {A1.value}\nPB {PB.value}\nM2 {Mc.value}\n"
            )
        )
        self.mDDGR = get_model(
            io.StringIO(
                f"{par}\nBINARY DDGR\nA1 {A1.value}\nPB {PB.value}\nM2 {Mc.value}\nMTOT {(Mp+Mc).value}\n"
            )
        )
        self.bDD = self.mDD.components["BinaryDD"].binary_instance
        self.bDDGR = self.mDDGR.components["BinaryDDGR"].binary_instance

    def test_pkparameters(self):
        pbdot = derived_quantities.pbdot(
            Mp, Mc, self.mDD.PB.quantity, self.mDD.ECC.quantity
        )
        gamma = derived_quantities.gamma(
            Mp, Mc, self.mDD.PB.quantity, self.mDD.ECC.quantity
        )
        omdot = derived_quantities.omdot(
            Mp, Mc, self.mDD.PB.quantity, self.mDD.ECC.quantity
        )
        assert np.isclose(gamma, self.bDDGR.GAMMA)
        assert np.isclose(pbdot, self.bDDGR.PBDOT)
        assert np.isclose(omdot, self.bDDGR._OMDOT)

    def test_binarydelay(self):
        # set the PK parameters
        self.mDD.GAMMA.value = self.bDDGR.GAMMA.value
        self.mDD.PBDOT.value = self.bDDGR.PBDOT.value
        self.mDD.OMDOT.value = self.bDDGR._OMDOT.value
        self.mDD.DR.value = self.bDDGR.DR.value
        self.mDD.DTH.value = self.bDDGR.DTH.value

        t = pint.simulation.make_fake_toas_uniform(55000, 57000, 100, model=self.mDD)
        DD_delay = self.mDD.binarymodel_delay(t, None)
        DDGR_delay = self.mDDGR.binarymodel_delay(t, None)
        assert np.allclose(DD_delay, DDGR_delay)
