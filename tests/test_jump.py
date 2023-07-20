"""Tests for jump model component """
import logging
import os
import pytest
import pytest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir
from pint.models import parameter as p
from pint.models import PhaseJump
import pint.models.timing_model
import pint.fitter


class SimpleSetup:
    def __init__(self, par, tim):
        self.par = par
        self.tim = tim
        self.m = mb.get_model(self.par)
        self.t = toa.get_TOAs(
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
    f = pint.fitter.Fitter.auto(setup_NGC6440E.t, setup_NGC6440E.m)

    # delete last jump
    setup_NGC6440E.m.delete_jump_and_flags(setup_NGC6440E.t.table["flags"], 2)
    for d in setup_NGC6440E.t.table["flags"][selected_toa_ind2]:
        assert "jump" not in d
    assert "PhaseJump" not in setup_NGC6440E.m.components
    f = pint.fitter.Fitter.auto(setup_NGC6440E.t, setup_NGC6440E.m)


def test_jump_params_to_flags(setup_NGC6440E):
    """Check jump_params_to_flags function."""
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
        assert dict["jump"] in ["1,2", "2,1"]
        assert dict["gui_jump"] == "2"
    assert len(cp.jumps) == 2

    setup_NGC6440E.m.delete_jump_and_flags(setup_NGC6440E.t.table["flags"], 2)
    for dict in setup_NGC6440E.t.table["flags"][selected_toa_ind]:
        assert "jump" in dict
    assert len(cp.jumps) == 1
    assert "JUMP1" in cp.jumps


def test_unfrozen_jump(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    # this has no TOAs
    par = p.maskParameter(
        name="JUMP", key="freq", value=0.2, key_value=[3000, 3200], units=u.s
    )
    setup_NGC6440E.m.components["PhaseJump"].add_param(par, setup=True)
    setup_NGC6440E.m.JUMP1.frozen = False
    with pytest.raises(pint.models.timing_model.MissingTOAs):
        setup_NGC6440E.m.validate_toas(setup_NGC6440E.t)


def test_find_empty_masks(setup_NGC6440E):
    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    # this has no TOAs
    par = p.maskParameter(
        name="JUMP", key="freq", value=0.2, key_value=[3000, 3200], units=u.s
    )
    setup_NGC6440E.m.components["PhaseJump"].add_param(par, setup=True)
    setup_NGC6440E.m.JUMP1.frozen = False
    bad_parameters = setup_NGC6440E.m.find_empty_masks(setup_NGC6440E.t)
    assert "JUMP1" in bad_parameters
    bad_parameters = setup_NGC6440E.m.find_empty_masks(setup_NGC6440E.t, freeze=True)
    setup_NGC6440E.m.validate_toas(setup_NGC6440E.t)


class TestJUMP:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parf = "B1855+09_NANOGrav_dfg+12_TAI.par"
        cls.timf = "B1855+09_NANOGrav_dfg+12.tim"
        cls.JUMPm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(
            cls.timf, ephem="DE405", planets=False, include_bipm=False
        )
        # libstempo calculation
        cls.ltres = np.genfromtxt(
            f"{cls.parf}.tempo_test", unpack=False, names=True, dtype=np.longdouble
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
        log.debug("Runing derivative for %s", f"d_delay_d_{p}")
        ndf = self.JUMPm.d_phase_d_param_num(self.toas, p)
        adf = self.JUMPm.d_phase_d_param(self.toas, self.JUMPm.delay(self.toas), p)
        diff = adf - ndf
        if np.all(diff.value) != 0.0:
            mean_der = (adf + ndf) / 2.0
            relative_diff = np.abs(diff) / np.abs(mean_der)
            # print "Diff Max is :", np.abs(diff).max()
            msg = (
                "Derivative test failed at d_phase_d_%s with max relative difference %lf"
                % (p, np.nanmax(relative_diff).value)
            )
            assert np.nanmax(relative_diff) < 0.001, msg
