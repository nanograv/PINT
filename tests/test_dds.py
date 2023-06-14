import os
import pytest
import io
import copy

import astropy.units as u
import numpy as np

from pint.models import get_model, get_model_and_toas
import pint.simulation
import pint.fitter
import pint.toa as toa
from pint import binaryconvert
from pinttestdata import datadir


def test_DDS_delay():
    """Make a copy of a DD model and switch to DDS"""
    parfileB1855 = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_modified.par")
    timB1855 = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim")
    t = toa.get_TOAs(timB1855, ephem="DE405", planets=False, include_bipm=False)
    mDD = get_model(parfileB1855)
    with open(parfileB1855) as f:
        lines = f.readlines()
    outlines = ""
    for line in lines:
        if not (line.startswith("SINI") or line.startswith("BINARY")):
            outlines += f"{line}"
        elif line.startswith("SINI"):
            d = line.split()
            sini = float(d[1])
            shapmax = -np.log(1 - sini)
            outlines += f"SHAPMAX {shapmax}\n"
        elif line.startswith("BINARY"):
            outlines += "BINARY DDS\n"
    mDDS = get_model(io.StringIO(outlines))
    DD_delay = mDD.binarymodel_delay(t, None)
    DDS_delay = mDDS.binarymodel_delay(t, None)
    assert np.allclose(DD_delay, DDS_delay)


class TestDDSFit:
    def setup_method(self):
        par = """
        PSR                            J1640+2224
        EPHEM                               DE440
        CLOCK                        TT(BIPM2019)
        UNITS                                 TDB
        START              53420.4178610467357292
        FINISH             59070.9677620557378125
        INFO                                   -f
        TIMEEPH                              FB90
        T2CMETHOD                        IAU2000B
        DILATEFREQ                              N
        DMDATA                                  N
        NTOA                                14035
        CHI2                    13829.35865041346
        ELONG                 243.989094207621264 0 0.00000001050784977998
        ELAT                   44.058509693269293 0 0.00000001093755261092
        PMELONG                 4.195360836354164 0 0.005558385201250641
        PMELAT                -10.720376071408817 0 0.008030254654447103
        PX                     0.5481471496073655 0 0.18913455936325857
        ECL                              IERS2010
        POSEPOCH           56246.0000000000000000
        F0                  316.12398420312142658 0 4.146144381519702124e-13
        F1              -2.8153754102544694156e-16 0 1.183507348296795048e-21
        PEPOCH             56246.0000000000000000
        CORRECT_TROPOSPHERE                         N
        PLANET_SHAPIRO                          N
        NE_SW                                 0.0
        SWM                                   0.0
        DM                   18.46722842706362534
        BINARY DD
        PB                   175.4606618995254901 1 3.282945138073456217e-09
        PBDOT                                 0.0
        A1                      55.32972013704859 1 1.4196834101027513e-06
        A1DOT              1.2273117569157312e-14 1 4.942065359144467e-16
        ECC                 0.0007972619266478279 1 7.475955296127427e-09
        EDOT              -5.4186414842257534e-17 1 1.7814213642177264e-17
        T0                 56188.1561739592320315 1 0.00021082819203119008413
        OM                  50.731718036317336052 1 0.00043256337136868348425
        OMDOT                                 0.0
        M2                    0.45422292168060424 1 0.17365015590922117
        SINI                   0.8774306786074643 1 0.048407859962533314
        A0                                    0.0
        B0                                    0.0
        GAMMA                                 0.0
        DR                                    0.0
        DTH                                   0.0
        """

        self.m = get_model(io.StringIO(par))
        self.mDDS = binaryconvert.convert_binary(self.m, "DDS")
        # use a specific seed for reproducible results
        np.random.seed(12345)
        self.t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, self.m, error=0.1 * u.us, add_noise=True
        )

    def test_resids(self):
        f = pint.fitter.Fitter.auto(self.t, self.m)
        fDDS = pint.fitter.Fitter.auto(self.t, self.mDDS)
        assert np.allclose(f.resids.time_resids, fDDS.resids.time_resids)

    def test_ddsfit(self):
        f = pint.fitter.Fitter.auto(self.t, self.m)
        f.fit_toas()
        chi2 = f.resids.calc_chi2()
        fDDS = pint.fitter.Fitter.auto(self.t, self.mDDS)

        fDDS.fit_toas()
        chi2DDS = fDDS.resids.calc_chi2()
        assert np.isclose(
            1 - np.exp(-fDDS.model.SHAPMAX.value), f.model.SINI.value, rtol=1e-2
        )
        print(f"{chi2} {chi2DDS}")
        assert np.isclose(chi2, chi2DDS, rtol=1e-2)

    def test_ddsfit_newSHAPMAX(self):
        f = pint.fitter.Fitter.auto(self.t, self.m)
        f.fit_toas()
        chi2 = f.resids.calc_chi2()
        fDDS = pint.fitter.Fitter.auto(self.t, self.mDDS)
        fDDS.model.SHAPMAX.quantity += 0.5
        fDDS.fit_toas()
        chi2DDS = fDDS.resids.calc_chi2()
        assert np.isclose(
            1 - np.exp(-fDDS.model.SHAPMAX.value), f.model.SINI.value, rtol=1e-2
        )
        assert np.isclose(chi2, chi2DDS, rtol=1e-2)
