from astropy import units as u, constants as c
import numpy as np
import io
import os

from pint.models import get_model
from pint import derived_quantities
import pint.simulation

# import pint.logging

# pint.logging.setup("WARNING")

# tests/datafile/B1855+09_NANOGrav_dfg+12_modified.par
# but with BINARY, SINI removed
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
PB             12.327171194774200418     0  0.00000000079493185824
T0             49452.940695077335647     0  0.00169031830532837251
A1             9.2307804312998001928     0  0.00000036890718667634
OM             276.55142180589701234     0  0.04936551005019605698
ECC            2.1745265668236919017e-05 0  0.00000004027191312623
M2             0.26111312480723428917    0  0.02616161008932908066
FD1  1.61666384D-04  1      3.38650356D-05
FD2 -1.88210030D-04  1      4.13173074D-05
FD3  1.07526915D-04  1      2.50177766D-05
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


def test_pkparameters():
    Mp = 1.4 * u.Msun
    mDD = get_model(
        io.StringIO(
            par
            + "\nBINARY DD\nSINI 0.99741717335200923866    0  0.00182023515130851988\n"
        )
    )

    Mc = mDD.M2.quantity
    mDDGR = get_model(
        io.StringIO(f"{par}\nBINARY DDGR\nMTOT {(Mp + mDD.M2.quantity).value}\n")
    )
    bDD = mDD.components["BinaryDD"].binary_instance
    bDDGR = mDDGR.components["BinaryDDGR"].binary_instance
    # reset SINI in the DD model to agree with GR
    mDD.SINI.quantity = bDDGR.SINI
    # check that we did it right
    Mc2 = derived_quantities.companion_mass(
        mDD.PB.quantity, mDD.A1.quantity, i=np.arcsin(mDD.SINI.quantity), mp=Mp
    )
    assert np.isclose(Mc, Mc2)
    pbdot = derived_quantities.pbdot(Mp, Mc, mDD.PB.quantity, mDD.ECC.quantity)
    gamma = derived_quantities.gamma(Mp, Mc, mDD.PB.quantity, mDD.ECC.quantity)
    omdot = derived_quantities.omdot(Mp, Mc, mDD.PB.quantity, mDD.ECC.quantity)
    assert np.isclose(gamma.value, bDDGR.GAMMA.value)
    assert np.isclose(pbdot.value, bDDGR.PBDOT.value)
    assert np.isclose(omdot.value, bDDGR._OMDOT.value)


def test_binarydelay():
    Mp = 1.4 * u.Msun
    mDD = get_model(
        io.StringIO(
            par
            + "\nBINARY DD\nSINI 0.99741717335200923866    0  0.00182023515130851988\n"
        )
    )

    Mc = mDD.M2.quantity
    mDDGR = get_model(
        io.StringIO(f"{par}\nBINARY DDGR\nMTOT {(Mp + mDD.M2.quantity).value}\n")
    )
    bDDGR = mDDGR.components["BinaryDDGR"].binary_instance
    # reset SINI in the DD model to agree with GR
    mDD.SINI.quantity = bDDGR.SINI
    pbdot = derived_quantities.pbdot(Mp, Mc, mDD.PB.quantity, mDD.ECC.quantity)
    gamma = derived_quantities.gamma(Mp, Mc, mDD.PB.quantity, mDD.ECC.quantity)
    omdot = derived_quantities.omdot(Mp, Mc, mDD.PB.quantity, mDD.ECC.quantity)

    # set the PK parameters
    mDD.GAMMA.value = gamma.value
    mDD.PBDOT.value = pbdot.value
    mDD.OMDOT.value = omdot.value

    t = pint.simulation.make_fake_toas_uniform(55000, 57000, 100, model=mDD)
    DD_delay = mDD.binarymodel_delay(t, None)
    DDGR_delay = mDDGR.binarymodel_delay(t, None)
    assert np.allclose(DD_delay, DDGR_delay)
