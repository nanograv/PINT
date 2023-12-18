import numpy as np
from astropy import units as u, constants as c
from pint.models import get_model_and_toas, get_model
import pint.fitter
import pint.logging
import os
import io
from pinttestdata import datadir
import pint.binaryconvert
import pint.derived_quantities
import pint.simulation

parDD = """
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
EDOT       2e-10 1 2e-12
"""
Mp = 1.4 * u.Msun
Mc = 1.1 * u.Msun
i = 85 * u.deg
PB = 0.5 * u.day
A1 = pint.derived_quantities.a1sini(Mp, Mc, PB, i)


def test_ddh_real():
    m, t = get_model_and_toas(
        os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"),
        os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"),
    )

    m2 = pint.binaryconvert.convert_binary(m, "DDH")

    f = pint.fitter.Fitter.auto(t, m)
    f2 = pint.fitter.Fitter.auto(t, m2)
    f.fit_toas()
    f2.fit_toas()
    assert np.isclose(f.resids.calc_chi2(), f2.resids.calc_chi2(), atol=0.5)
    assert np.isclose(f.model.M2.value, f2.model.M2.value, atol=0.01)
    assert np.isclose(f.model.SINI.value, f2.model.SINI.value, atol=0.01)


def test_ddh_sim():
    m = get_model(
        io.StringIO(
            f"{parDD}\nBINARY DD\nSINI {np.sin(i).value}\nA1 {A1.value}\nPB {PB.value}\nM2 {Mc.value}\n"
        )
    )
    t = pint.simulation.make_fake_toas_uniform(50000, 51000, 1000, m, add_noise=True)
    m2 = pint.binaryconvert.convert_binary(m, "DDH")
    for p in m.free_params:
        if p not in ["M2", "SINI"]:
            getattr(m, p).frozen = True
    for p in m2.free_params:
        if p not in ["H3", "STIGMA"]:
            getattr(m2, p).frozen = True

    f = pint.fitter.Fitter.auto(t, m)
    f2 = pint.fitter.Fitter.auto(t, m2)
    f.fit_toas()
    f2.fit_toas()
    assert np.isclose(f.resids.calc_chi2(), f2.resids.calc_chi2(), atol=0.5)
    assert np.isclose(f.model.M2.value, f2.model.M2.value, atol=0.01)
    assert np.isclose(f.model.SINI.value, f2.model.SINI.value, atol=0.01)
