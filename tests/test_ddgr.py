from astropy import units as u, constants as c
import numpy as np
import io
import os
import copy

from pint.models import get_model
from pint import derived_quantities
import pint.simulation
import pint.fitter

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
i = 75 * u.deg
PB = 0.5 * u.day


class TestDDGR:
    def setup_method(self):
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
        assert np.isclose(gamma, self.mDDGR.GAMMA.quantity)
        assert np.isclose(pbdot, self.mDDGR.PBDOT.quantity)
        assert np.isclose(omdot, self.mDDGR.OMDOT.quantity)

    def test_binarydelay(self):
        # set the PK parameters
        self.mDD.GAMMA.value = self.mDDGR.GAMMA.value
        self.mDD.PBDOT.value = self.mDDGR.PBDOT.value
        self.mDD.OMDOT.value = self.mDDGR.OMDOT.value
        self.mDD.DR.value = self.mDDGR.DR.value
        self.mDD.DTH.value = self.mDDGR.DTH.value

        t = pint.simulation.make_fake_toas_uniform(55000, 57000, 100, model=self.mDD)
        DD_delay = self.mDD.binarymodel_delay(t, None)
        DDGR_delay = self.mDDGR.binarymodel_delay(t, None)
        assert np.allclose(DD_delay, DDGR_delay)

    def test_xomdot(self):
        self.mDD.GAMMA.value = self.mDDGR.GAMMA.value
        self.mDD.PBDOT.value = self.mDDGR.PBDOT.value
        self.mDD.OMDOT.value = self.mDDGR.OMDOT.value * 2
        self.mDD.DR.value = self.mDDGR.DR.value
        self.mDD.DTH.value = self.mDDGR.DTH.value

        self.mDDGR.XOMDOT.value = self.mDDGR.OMDOT.value
        t = pint.simulation.make_fake_toas_uniform(55000, 57000, 100, model=self.mDD)
        DD_delay = self.mDD.binarymodel_delay(t, None)
        DDGR_delay = self.mDDGR.binarymodel_delay(t, None)
        assert np.allclose(DD_delay, DDGR_delay)

    def test_xpbdot(self):
        self.mDD.GAMMA.value = self.mDDGR.GAMMA.value
        self.mDD.PBDOT.value = self.mDDGR.PBDOT.value * 2
        self.mDD.OMDOT.value = self.mDDGR.OMDOT.value
        self.mDD.DR.value = self.mDDGR.DR.value
        self.mDD.DTH.value = self.mDDGR.DTH.value

        self.mDDGR.XPBDOT.value = self.mDDGR.PBDOT.value
        t = pint.simulation.make_fake_toas_uniform(55000, 57000, 100, model=self.mDD)
        DD_delay = self.mDD.binarymodel_delay(t, None)
        DDGR_delay = self.mDDGR.binarymodel_delay(t, None)
        assert np.allclose(DD_delay, DDGR_delay)

    def test_ddgrfit_noMTOT(self):
        # set the PK parameters
        self.mDD.GAMMA.value = self.mDDGR.GAMMA.value
        self.mDD.PBDOT.value = self.mDDGR.PBDOT.value
        self.mDD.OMDOT.value = self.mDDGR.OMDOT.value
        self.mDD.DR.value = self.mDDGR.DR.value
        self.mDD.DTH.value = self.mDDGR.DTH.value

        t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, error=1 * u.us, add_noise=True, model=self.mDD
        )

        fDD = pint.fitter.Fitter.auto(t, self.mDD)
        fDDGR = pint.fitter.Fitter.auto(t, self.mDDGR)
        for p in ["ECC", "PB", "A1", "OM", "T0"]:
            getattr(fDD.model, p).frozen = False
            getattr(fDDGR.model, p).frozen = False

        fDD.model.GAMMA.frozen = False
        fDD.model.PBDOT.frozen = False
        fDD.model.OMDOT.frozen = False
        fDD.model.SINI.frozen = False
        fDD.model.M2.frozen = False

        # cannot fit for MTOT yet
        fDDGR.model.M2.frozen = False
        fDDGR.model.MTOT.frozen = True
        fDD.fit_toas()
        chi2DD = fDD.resids.calc_chi2()

        fDDGR.fit_toas()
        chi2DDGR = fDDGR.resids.calc_chi2()
        M2 = copy.deepcopy(fDDGR.model.M2.quantity)
        # chi^2 values don't have to be super close
        assert (
            np.fabs(fDD.model.M2.quantity - fDDGR.model.M2.quantity)
            < 5 * fDD.model.M2.uncertainty
        )
        # perturn M2 and make sure chi^2 gets worse
        fDDGR.model.M2.quantity += 3 * fDDGR.model.M2.uncertainty
        fDDGR.resids.update()
        assert fDDGR.resids.calc_chi2() > chi2DDGR
        fDDGR.fit_toas()
        assert np.isclose(fDDGR.resids.calc_chi2(), chi2DDGR, atol=0.1)
        assert np.isclose(fDDGR.model.M2.quantity, M2, atol=1e-7)

    def test_ddgrfit(self):
        t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, model=self.mDDGR, error=1 * u.us, add_noise=True
        )
        fDDGR = pint.fitter.Fitter.auto(t, self.mDDGR)

        fDDGR.model.M2.frozen = False
        fDDGR.model.MTOT.frozen = False
        # mDDGR.XOMDOT.frozen = False
        fDDGR.model.XPBDOT.frozen = False

        # start well away from best-fit
        fDDGR.model.MTOT.quantity += 1e-4 * u.Msun
        fDDGR.model.M2.quantity += 1e-2 * u.Msun
        fDDGR.update_resids()

        fDDGR.fit_toas()
        assert (
            np.abs(fDDGR.model.MTOT.quantity - (Mp + Mc))
            < 4 * fDDGR.model.MTOT.uncertainty
        )
        assert np.abs(fDDGR.model.M2.quantity - (Mc)) < 4 * fDDGR.model.M2.uncertainty
        assert np.abs(fDDGR.model.XPBDOT.quantity) < 4 * fDDGR.model.XPBDOT.uncertainty

    def test_design_XOMDOT(self):
        t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, model=self.mDDGR, error=1 * u.us, add_noise=True
        )
        f = pint.fitter.Fitter.auto(t, self.mDDGR)
        for p in f.model.free_params:
            getattr(f.model, p).frozen = True
        f.model.XOMDOT.frozen = False
        f.model.XOMDOT.value = 0
        f.fit_toas()
        XOMDOT = 0 * u.deg / u.yr
        dXOMDOT = 1e-6 * u.deg / u.yr
        # move away from minimum
        f.model.XOMDOT.quantity = XOMDOT + dXOMDOT
        f.update_resids()
        M, pars, units = f.model.designmatrix(f.toas)
        # this is recalculating chi^2 for comparison
        chi2start = (
            ((f.resids.calc_time_resids() / f.toas.get_errors()).decompose()) ** 2
        ).sum()
        chi2pred = (
            (
                (
                    (f.resids.calc_time_resids() - M[:, 1] * dXOMDOT.value / u.Hz)
                    / f.toas.get_errors()
                ).decompose()
            )
            ** 2
        ).sum()
        f.model.XOMDOT.quantity = XOMDOT + dXOMDOT * 2
        f.update_resids()
        chi2found = f.resids.calc_chi2()
        assert np.isclose(chi2pred, chi2found, rtol=1e-2)

    def test_design_XPBDOT(self):
        t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, model=self.mDDGR, error=1 * u.us, add_noise=True
        )
        f = pint.fitter.Fitter.auto(t, self.mDDGR)
        for p in f.model.free_params:
            getattr(f.model, p).frozen = True
        f.model.XPBDOT.frozen = False
        f.model.XPBDOT.value = 0
        f.fit_toas()
        XPBDOT = 0 * u.s / u.s
        dXPBDOT = 1e-14 * u.s / u.s
        # move away from minimum
        f.model.XPBDOT.quantity = XPBDOT + dXPBDOT
        f.update_resids()
        M, pars, units = f.model.designmatrix(f.toas)
        # this is recalculating chi^2 for comparison
        chi2start = (
            ((f.resids.calc_time_resids() / f.toas.get_errors()).decompose()) ** 2
        ).sum()
        chi2pred = (
            (
                (
                    (f.resids.calc_time_resids() - M[:, 1] * dXPBDOT.value / u.Hz)
                    / f.toas.get_errors()
                ).decompose()
            )
            ** 2
        ).sum()
        f.model.XPBDOT.quantity = XPBDOT + dXPBDOT * 2
        f.update_resids()
        chi2found = f.resids.calc_chi2()
        assert np.isclose(chi2pred, chi2found, rtol=1e-2)

    def test_design_M2(self):
        t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, model=self.mDDGR, error=1 * u.us, add_noise=True
        )
        f = pint.fitter.Fitter.auto(t, self.mDDGR)
        for p in f.model.free_params:
            getattr(f.model, p).frozen = True
        f.model.M2.frozen = False
        f.fit_toas()
        M2 = f.model.M2.quantity
        dM2 = 1e-4 * u.Msun
        # move away from minimum
        f.model.M2.quantity = M2 + dM2
        f.update_resids()
        M, pars, units = f.model.designmatrix(f.toas)
        # this is recalculating chi^2 for comparison
        chi2start = (
            ((f.resids.calc_time_resids() / f.toas.get_errors()).decompose()) ** 2
        ).sum()
        chi2pred = (
            (
                (
                    (f.resids.calc_time_resids() - M[:, 1] * dM2.value / u.Hz)
                    / f.toas.get_errors()
                ).decompose()
            )
            ** 2
        ).sum()
        f.model.M2.quantity = M2 + dM2 * 2
        f.update_resids()
        chi2found = f.resids.calc_chi2()
        assert np.isclose(chi2pred, chi2found, rtol=1e-2)

    def test_design_MTOT(self):
        t = pint.simulation.make_fake_toas_uniform(
            55000, 57000, 100, model=self.mDDGR, error=1 * u.us, add_noise=True
        )
        f = pint.fitter.Fitter.auto(t, self.mDDGR)
        for p in f.model.free_params:
            getattr(f.model, p).frozen = True
        f.model.MTOT.frozen = False
        f.fit_toas()
        MTOT = f.model.MTOT.quantity
        dMTOT = 1e-5 * u.Msun
        # move away from minimum
        f.model.MTOT.quantity = MTOT + dMTOT
        f.update_resids()
        M, pars, units = f.model.designmatrix(f.toas)
        # this is recalculating chi^2 for comparison
        chi2start = (
            ((f.resids.calc_time_resids() / f.toas.get_errors()).decompose()) ** 2
        ).sum()
        chi2pred = (
            (
                (
                    (f.resids.calc_time_resids() - M[:, 1] * dMTOT.value / u.Hz)
                    / f.toas.get_errors()
                ).decompose()
            )
            ** 2
        ).sum()
        f.model.MTOT.quantity = MTOT + dMTOT * 2
        f.update_resids()
        chi2found = f.resids.calc_chi2()
        assert np.isclose(chi2pred, chi2found, rtol=1e-2)
