"""Tests for jump model component """
import logging
import os
import re
import unittest
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pinttestdata import datadir

import pint.models.parameter
import pint.toa
from pint.models import PhaseJump, get_model, parameter as p
from pint.residuals import Residuals


class SimpleSetup:
    def __init__(self, par, tim):
        self.par = par
        self.tim = tim
        self.m = get_model(self.par)
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
    j2 = cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind2])
    jp2 = getattr(cp, j2)
    assert jp2.flag == "-gui_jump"
    assert jp2.flag_value == "2"
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
    j1 = cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind])
    j2 = cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind2])
    # attempt to add overlapping jump - should not add jump
    selected_toa_ind3 = [9, 10, 11]
    with pytest.raises(ValueError):
        cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind3])
    # check previous jump flags unaltered
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert d["gui_jump"] == "1"
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind2]:
        assert d["gui_jump"] == "2"
    # check that no flag added to index 9
    assert "jump" not in setup_NGC6440E.t.table[9].colnames
    assert "gui_jump" not in setup_NGC6440E.t.table[9].colnames


def test_multijump_toa(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]
    par = p.maskParameter(
        name="JUMP",
        flag="freq",
        value=0.2,
        flag_value=[1440 * u.MHz, 1700 * u.MHz],
        units=u.s,
    )  # TOAs indexed 48, 49, 54 in NGC6440E are within this frequency range
    selected_toa_ind = [48, 49, 54]
    cp.add_param(par, setup=True)

    # check that one can still add "gui jumps" to model-jumped TOAs
    cp.add_jump_and_flags(setup_NGC6440E.t.table["flags"][selected_toa_ind])
    assert_array_equal(setup_NGC6440E.t["gui_jump", selected_toa_ind], "1")
    assert len(cp.jumps) == 2


class TestJUMP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parf = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_TAI.par")
        cls.timf = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim")
        cls.JUMPm = get_model(cls.parf)
        cls.toas = pint.toa.get_TOAs(
            cls.timf, ephem="DE405", planets=False, include_bipm=False
        )
        # libstempo calculation
        cls.ltres = np.genfromtxt(
            cls.parf + ".tempo_test", unpack=False, names=True, dtype=np.longdouble
        )

    def test_jump_agrees_with_tempo(self):
        presids_s = Residuals(
            self.toas, self.JUMPm, use_weighted_mean=False
        ).time_resids.to(u.s)
        assert np.all(
            np.abs(presids_s.value - self.ltres["residuals"]) < 1e-7
        ), "JUMP test failed."

    def test_jump_selects_toas(self):
        for p in self.JUMPm.params:
            if not p.startswith("JUMP"):
                continue
            pm = getattr(self.JUMPm, p)
            if pm.flag != "-chanid":
                continue
            assert len(
                [l for l in open(self.timf).readlines() if re.search(pm.flag_value, l)]
            ) == len(pm.select_toa_mask(self.toas))

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


@pytest.mark.parametrize(
    "tim, flag_ranges",
    [
        (
            """
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
""",
            [(57100, 57250, 1), (57250, 57400, 2), (57600, 59000, 3)],
        ),
        (
            """
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
""",
            [(57100, 57250, 1), (57600, 59000, 3)],
        ),
    ],
)
def test_tim_file_gets_jump_flags(tim, flag_ranges):
    toas = pint.toa.get_TOAs(StringIO(tim))
    for start, end, n in flag_ranges:
        for i in range(len(toas)):
            f = toas.table["flags"][i]
            m = int(f.get("jump", -1))
            assert (start < toas.table["tdbld"][i] < end) == (n == m)


def test_multiple_jumps_add():
    m = get_model(
        StringIO(
            """
    PSR J1234+5678
    ELAT 0
    ELONG 0
    PEPOCH 57000
    POSEPOCH 57000
    F0 500
    JUMP mjd 58000 60000 0
    JUMP mjd 59000 60000 0
    """
        )
    )
    for jmp in m.jumps:
        if jmp.flag == "mjd":
            start, end = jmp.flag_value
            if start < 58500:
                first_jump = jmp
            else:
                second_jump = jmp
    toas = pint.toa.make_fake_toas(57000, 60000 - 1, 10, m)

    first_jump.quantity = 100 * u.us
    second_jump.quantity = 0 * u.us
    r_first = pint.residuals.Residuals(toas, m)

    first_jump.quantity = 0 * u.us
    second_jump.quantity = 75 * u.us
    r_second = pint.residuals.Residuals(toas, m)

    first_jump.quantity = 100 * u.us
    second_jump.quantity = 75 * u.us
    r_sum = pint.residuals.Residuals(toas, m)

    assert_allclose(
        (r_first.resids + r_second.resids).to_value(u.us),
        r_sum.resids.to_value(u.us),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "j",
    [
        pint.models.parameter.maskParameter(
            name="JUMP",
            flag="-fish",
            flag_value="carp",
            units=u.s,
            value=7,
            frozen=False,
            uncertainty=0.1,
        ),
        pint.models.parameter.maskParameter(
            name="JUMP", flag="tel", flag_value="ao", units=u.s, value=7, frozen=False,
        ),
        pint.models.parameter.maskParameter(
            name="JUMP",
            flag="MJD",
            flag_value=(57000, 58000,),
            units=u.s,
            value=7,
            frozen=False,
        ),
    ],
)
def test_jump_parfile_roundtrip(j):
    l = j.as_parfile_line()
    nj = pint.models.parameter.maskParameter(name="JUMP", units=u.s)
    nj.from_parfile_line(l)

    assert nj.flag == j.flag
    assert nj.flag_value == j.flag_value
    if nj.quantity != j.quantity:
        assert_allclose(nj.quantity, j.quantity)
    assert nj.frozen == j.frozen
    if nj.uncertainty != j.uncertainty:
        assert_allclose(nj.uncertainty, j.uncertainty)


@pytest.fixture
def small():
    m = get_model(
        StringIO(
            """
    PSR J1234+5678
    F0 1
    PEPOCH 58000
    POSEPOCH 58000
    ELONG 0
    ELAT 0
    JUMP mjd 59000 70000 0
    """
        )
    )
    t = pint.toa.make_fake_toas(58000, 60000, 20, m)

    class R:
        pass

    r = R()
    r.m = m
    r.t = t
    for j in m.jumps:
        if j.flag == "mjd":
            r.j = j
    return r


def test_tidy_jumps_all_ok(small):
    small.j.frozen = False
    small.m.tidy_jumps_for_fit(small.t)
    assert not small.j.frozen


def test_tidy_jumps_all_jumped(small):
    small.j.frozen = False
    small.j.flag_value = (
        56000,
        70000,
    )
    small.m.tidy_jumps_for_fit(small.t)
    assert small.j.frozen


def test_tidy_jumps_irrelevant(small):
    small.j.frozen = False
    j2 = small.j.new_param(index=100, copy_all=True)
    j2.flag_value = (
        50000,
        55000,
    )
    small.m.components["PhaseJump"].add_param(j2)
    small.m.tidy_jumps_for_fit(small.t)
    assert not small.j.frozen
    assert j2.frozen


def test_tidy_jumps_cover_all_freeze_one(small):
    small.j.frozen = False
    j2 = small.j.new_param(index=100, copy_all=True)
    j2.flag_value = (
        50000,
        59000,
    )
    small.m.components["PhaseJump"].add_param(j2)
    small.m.tidy_jumps_for_fit(small.t)
    assert small.j.frozen != j2.frozen
