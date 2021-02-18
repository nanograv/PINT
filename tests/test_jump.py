"""Tests for jump model component """
import logging
import os
import unittest
import pytest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir
from pint.models import parameter as p
from pint.models import PhaseJump


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
        assert setup_NGC6440E.t.table["flags"][i]["jump"] == 1
    # ensure no extraneous flags added to unaffected TOAs
    for i in range(setup_NGC6440E.t.ntoas):
        if i not in toa_indeces:
            assert "jump" not in setup_NGC6440E.t.table["flags"][i]

    # check case where multiple calls are performed (no-ops)
    old_table = setup_NGC6440E.t.table
    setup_NGC6440E.m.jump_params_to_flags(setup_NGC6440E.t)
    assert all(old_table) == all(setup_NGC6440E.t.table)


class TestJUMP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(datadir)
        cls.parf = "B1855+09_NANOGrav_dfg+12_TAI.par"
        cls.timf = "B1855+09_NANOGrav_dfg+12.tim"
        cls.JUMPm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(
            cls.timf, ephem="DE405", planets=False, include_bipm=False
        )
        # libstempo calculation
        cls.ltres = np.genfromtxt(
            cls.parf + ".tempo_test", unpack=True, names=True, dtype=np.longdouble
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


if __name__ == "__main__":
    pass
