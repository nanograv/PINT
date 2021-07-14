"""Tests for jump model component """
import logging
import os
import unittest
import pytest
from io import StringIO

import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

import pint.models.model_builder as mb
import pint.toa
from pint.residuals import Residuals
from pinttestdata import datadir
from pint.models import parameter as p
from pint.models import PhaseJump


class SimpleSetup:
    def __init__(self, par, tim):
        self.par = par
        self.tim = tim
        self.m = mb.get_model(self.par)
        self.t = pint.toa.get_TOAs(
            self.tim, ephem="DE405", planets=False, include_bipm=False
        )


@pytest.fixture
def setup_NGC6440E():
    os.chdir(datadir)
    return SimpleSetup("NGC6440E.par", "NGC6440E.tim")


def test_add_jumps_and_flags(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]

    # simulate selecting TOAs in pintk and jumping them
    selected_toa_ind = [1, 2, 3]  # arbitrary set of TOAs
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind])
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert d["gui_jump"] == "1"

    # add second jump to different set of TOAs
    selected_toa_ind2 = [10, 11, 12]
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind2])
    # check previous jump flags unaltered
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert d["gui_jump"] == "1"
    # check appropriate flags added
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind2]:
        assert d["gui_jump"] == "2"


def test_add_overlapping_jump(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]
    selected_toa_ind = [1, 2, 3]
    selected_toa_ind2 = [10, 11, 12]
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind])
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind2])
    # attempt to add overlapping jump - should not add jump
    selected_toa_ind3 = [9, 10, 11]
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind3])
    # check previous jump flags unaltered
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert d["gui_jump"] == "1"
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind2]:
        assert d["gui_jump"] == "2"
    # check that no flag added to index 9
    assert "jump" not in setup_NGC6440E.t.table[9].colnames
    assert "gui_jump" not in setup_NGC6440E.t.table[9].colnames


def test_remove_jump_and_flags(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]
    selected_toa_ind = [1, 2, 3]
    selected_toa_ind2 = [10, 11, 12]
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind])
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind2])
    # test delete_jump_and_flags
    setup_NGC6440E.m.delete_jump_and_flags(setup_NGC6440E.t.table["flags"], 1)
    assert len(cp.jumps) == 1

    # delete last jump
    setup_NGC6440E.m.delete_jump_and_flags(setup_NGC6440E.t.table["flags"], 2)
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind2]:
        assert "jump" not in d
    assert "PhaseJump" not in setup_NGC6440E.m.components


def test_jump_params_to_flags(setup_NGC6440E):
    """ Check jump_params_to_flags function. """
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]

    par = p.maskParameter(
        name="JUMP", key="freq", value=0.2, key_value=[1440, 1700], units=u.s
    )  # TOAs indexed 48, 49, 54 in NGC6440E are within this frequency range
    cp.add_param(par, setup=True)

    # sanity check - ensure no jump flags from initialization
    for i in range(setup_NGC6440E.t.ntoas):
        assert "jump" not in setup_NGC6440E.t.table["flags"][i]

    # add flags based off jumps added to model
    setup_NGC6440E.m.jump_params_to_flags(setup_NGC6440E.t)

    # index to affected TOAs and ensure appropriate flags set
    toa_indeces = [48, 49, 54]
    for i in toa_indeces:
        assert "jump" in setup_NGC6440E.t.table["flags"][i]
        assert setup_NGC6440E.t.table["flags"][i]["jump"][0] == "1"
    # ensure no extraneous flags added to unaffected TOAs
    for i in range(setup_NGC6440E.t.ntoas):
        if i not in toa_indeces:
            assert "jump" not in setup_NGC6440E.t.table["flags"][i]

    # check case where multiple calls are performed (no-ops)
    old_table = setup_NGC6440E.t.table
    setup_NGC6440E.m.jump_params_to_flags(setup_NGC6440E.t)
    assert all(old_table) == all(setup_NGC6440E.t.table)

    # check that adding overlapping jump works
    par2 = p.maskParameter(
        name="JUMP", key="freq", value=0.2, key_value=[1600, 1900], units=u.s
    )  # frequency range overlaps with par, 2nd jump will have common TOAs w/ 1st
    cp.add_param(par2, setup=True)
    # add flags based off jumps added to model
    setup_NGC6440E.m.jump_params_to_flags(setup_NGC6440E.t)
    mask2 = par2.select_toa_mask(setup_NGC6440E.t)
    intersect = np.intersect1d(toa_indeces, mask2)
    assert intersect is not []
    for i in mask2:
        assert "2" in setup_NGC6440E.t.table["flags"][i]["jump"]
    for i in toa_indeces:
        assert "1" in setup_NGC6440E.t.table["flags"][i]["jump"]


def test_multijump_toa(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]
    par = p.maskParameter(
        name="JUMP", key="freq", value=0.2, key_value=[1440, 1700], units=u.s
    )  # TOAs indexed 48, 49, 54 in NGC6440E are within this frequency range
    selected_toa_ind = [48, 49, 54]
    cp.add_param(par, setup=True)

    # check that one can still add "gui jumps" to model-jumped TOAs
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind])
    # add flags based off jumps added to model
    setup_NGC6440E.m.jump_params_to_flags(setup_NGC6440E.t)
    for dict in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert dict["jump"] == "1,2"
        assert dict["gui_jump"] == "2"
    assert len(cp.jumps) == 2

    setup_NGC6440E.m.delete_jump_and_flags(setup_NGC6440E.t.table["flags"], 2)
    for dict in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert "jump" in dict
    assert len(cp.jumps) == 1
    assert "JUMP1" in cp.jumps


class TestJUMP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parf = "B1855+09_NANOGrav_dfg+12_TAI.par"
        cls.timf = "B1855+09_NANOGrav_dfg+12.tim"
        cls.JUMPm = mb.get_model(cls.parf)
        cls.toas = pint.toa.get_TOAs(
            cls.timf, ephem="DE405", planets=False, include_bipm=False
        )
        # libstempo calculation
        cls.ltres = np.genfromtxt(
            cls.parf + ".tempo_test", unpack=False, names=True, dtype=np.longdouble
        )

    def test_jump(self):
        presids_s = Residuals(
            self.toas, self.JUMPm, use_weighted_mean=False
        ).time_resids.to(u.s)
        assert np.all(
            np.abs(presids_s.value - self.ltres["residuals"]) < 1e-7
        ), "JUMP test failed."

    def test_derivative(self):
        log = logging.getLogger("Jump phase test")
        p = "JUMP2"
        log.debug("Runing derivative for %s", "d_delay_d_" + p)
        ndf = self.JUMPm.d_phase_d_param_num(self.toas, p)
        adf = self.JUMPm.d_phase_d_param(self.toas, self.JUMPm.delay(self.toas), p)
        diff = adf - ndf
        if not np.all(diff.value) == 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_phase_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            assert np.nanmax(relative_diff) < 0.001, msg

@pytest.mark.parametrize("tim, flag_ranges", [
("""
FORMAT 1
unk 999999.000000 57000.0000000078830324 1.000 gbt  -pn -2273593021.0
unk 999999.000000 57052.6315789538116088 1.000 gbt  -pn -124285.0
JUMP
unk 999999.000000 57105.2631578955917593 1.000 gbt  -pn 2273470219.0
unk 999999.000000 57157.8947368420341204 1.000 gbt  -pn 4547256435.0
unk 999999.000000 57210.5263157897339815 1.000 gbt  -pn 6821152750.0
JUMP
JUMP
unk 999999.000000 57263.1578947318551852 1.000 gbt  -pn 9095000767.0
unk 999999.000000 57315.7894736865498264 1.000 gbt  -pn 11368677703.0
unk 999999.000000 57368.4210526391189699 1.000 gbt  -pn 13642183645.0
JUMP
unk 999999.000000 57421.0526315687085764 1.000 gbt  -pn 15915655534.0
unk 999999.000000 57473.6842105144226621 1.000 gbt  -pn 18189261252.0
unk 999999.000000 57526.3157894708454977 1.000 gbt  -pn 20463057581.0
unk 999999.000000 57578.9473684237653588 1.000 gbt  -pn 22736955090.0
JUMP
unk 999999.000000 57631.5789473769360416 1.000 gbt  -pn 25010795383.0
unk 999999.000000 57684.2105263208505671 1.000 gbt  -pn 27284460486.0
unk 999999.000000 57736.8421052672976158 1.000 gbt  -pn 29557959408.0
unk 999999.000000 57789.4736842041808449 1.000 gbt  -pn 31831435551.0
unk 999999.000000 57842.1052631637698032 1.000 gbt  -pn 34105052652.0
unk 999999.000000 57894.7368420936182176 1.000 gbt  -pn 36378858766.0
unk 999999.000000 57947.3684210606924768 1.000 gbt  -pn 38652757406.0
unk 999999.000000 57999.9999999883338542 1.000 gbt  -pn 40926589498.0
JUMP
""", [(57100, 57250, 1), (57250, 57400, 2), (57600, 59000, 3)]),
("""
FORMAT 1
unk 999999.000000 57000.0000000078830324 1.000 gbt  -pn -2273593021.0
unk 999999.000000 57052.6315789538116088 1.000 gbt  -pn -124285.0
JUMP
unk 999999.000000 57105.2631578955917593 1.000 gbt  -pn 2273470219.0
unk 999999.000000 57157.8947368420341204 1.000 gbt  -pn 4547256435.0
unk 999999.000000 57210.5263157897339815 1.000 gbt  -pn 6821152750.0
JUMP
unk 999999.000000 57263.1578947318551852 1.000 gbt  -pn 9095000767.0
unk 999999.000000 57315.7894736865498264 1.000 gbt  -pn 11368677703.0
unk 999999.000000 57368.4210526391189699 1.000 gbt  -pn 13642183645.0
JUMP
JUMP
unk 999999.000000 57421.0526315687085764 1.000 gbt  -pn 15915655534.0
unk 999999.000000 57473.6842105144226621 1.000 gbt  -pn 18189261252.0
unk 999999.000000 57526.3157894708454977 1.000 gbt  -pn 20463057581.0
unk 999999.000000 57578.9473684237653588 1.000 gbt  -pn 22736955090.0
JUMP
unk 999999.000000 57631.5789473769360416 1.000 gbt  -pn 25010795383.0
unk 999999.000000 57684.2105263208505671 1.000 gbt  -pn 27284460486.0
unk 999999.000000 57736.8421052672976158 1.000 gbt  -pn 29557959408.0
unk 999999.000000 57789.4736842041808449 1.000 gbt  -pn 31831435551.0
unk 999999.000000 57842.1052631637698032 1.000 gbt  -pn 34105052652.0
unk 999999.000000 57894.7368420936182176 1.000 gbt  -pn 36378858766.0
unk 999999.000000 57947.3684210606924768 1.000 gbt  -pn 38652757406.0
unk 999999.000000 57999.9999999883338542 1.000 gbt  -pn 40926589498.0
JUMP
""", [(57100, 57250, 1), (57600, 59000, 3)]),
])
def test_tim_file_gets_jump_flags(tim, flag_ranges):
    toas = pint.toa.get_TOAs(StringIO(tim))
    for start, end, n in flag_ranges:
        for i in range(len(toas)):
            f = toas.table["flags"][i]
            m = int(f.get("jump", -1))
            assert (start < toas.table["tdbld"][i] < end) == (n == m)

def test_multiple_jumps_add():
    m = mb.get_model(StringIO("""
    PSR J1234+5678
    ELAT 0
    ELONG 0
    PEPOCH 57000
    POSEPOCH 57000
    F0 500
    JUMP mjd 58000 60000 0
    JUMP mjd 59000 60000 0
    """))
    for j in m.jumps:
        jmp = getattr(m, j)
        if jmp.key == 'mjd':
            start, end = jmp.key_value
            if start < 58500:
                first_jump = jmp
            else:
                second_jump = jmp
    toas = pint.toa.make_fake_toas(57000, 60000-1, 10, m)

    first_jump.quantity = 100*u.us
    second_jump.quantity = 0*u.us
    r_first = pint.residuals.Residuals(toas, m)

    first_jump.quantity = 0*u.us
    second_jump.quantity = 75*u.us
    r_second = pint.residuals.Residuals(toas, m)

    first_jump.quantity = 100*u.us
    second_jump.quantity = 75*u.us
    r_sum = pint.residuals.Residuals(toas, m)

    assert_allclose(r_first.resids + r_second.resids, r_sum.resids, atol=1e-3*u.us)
