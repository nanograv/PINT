"""Tests for grouping and flagging"""
import logging
import os
import unittest
import pytest
import copy

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


def test_add_flags(setup_NGC6440E):
    """
    test adding a new flag
    """
    setup_NGC6440E.t.add_flags(np.arange(len(setup_NGC6440E.t)), "newflag", "test")
    assert np.array(["newflag" in x for x in setup_NGC6440E.t.get_flags()]).all()
    flag_values, flag_indices = setup_NGC6440E.t.get_flag_value("newflag")
    assert (np.array(flag_values) == "test").all()


def test_add_flag_twice(setup_NGC6440E):
    """
    test adding a new flag twice - should raise exception
    """
    setup_NGC6440E.t.add_flags(np.arange(len(setup_NGC6440E.t)), "newflag", "test")
    with pytest.raises(ValueError):
        setup_NGC6440E.t.add_flags(np.arange(len(setup_NGC6440E.t)), "newflag", "test")


def test_add_flag_twice_overwrite(setup_NGC6440E):
    """
    test adding a new flag twice but with overwriting specified - should not raise exception
    """
    setup_NGC6440E.t.add_flags(np.arange(len(setup_NGC6440E.t)), "newflag", "test")
    setup_NGC6440E.t.add_flags(
        np.arange(len(setup_NGC6440E.t)), "newflag", "test2", overwrite=True
    )
    flag_values, flag_indices = setup_NGC6440E.t.get_flag_value("newflag")
    assert (np.array(flag_values) == "test2").all()


def test_jump_by_group(setup_NGC6440E):
    """
    Compare selection by MJD with selection by group (which by default selects TOAs by 2h gaps)
    """
    m_copy = copy.deepcopy(setup_NGC6440E.m)

    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]
    par = p.maskParameter(
        name="JUMP", key="mjd", value=0.2, key_value=[54099, 54100], units=u.s
    )
    # this should be the last group of any TOAs
    cp.add_param(par, setup=True)

    # add groups to the TOAs
    groups = setup_NGC6440E.t.get_groups(2 * u.hr, add_column=False)
    # convert group to string
    groups = [str(x) for x in groups]
    setup_NGC6440E.t.add_flags(np.arange(len(setup_NGC6440E.t)), "toagroup", groups)

    m_copy.add_component(PhaseJump(), validate=False)
    cp_copy = m_copy.components["PhaseJump"]
    par_copy = p.maskParameter(
        name="JUMP", key="-toagroup", value=0.2, key_value=41, units=u.s
    )
    # this should be identical to the group above
    cp_copy.add_param(par_copy, setup=True)
    assert (
        np.array(cp.JUMP1.select_toa_mask(setup_NGC6440E.t))
        == np.array(cp_copy.JUMP1.select_toa_mask(setup_NGC6440E.t))
    ).all(), (
        "%s vs. %s"
        % (
            cp.JUMP1.select_toa_mask(setup_NGC6440E.t),
            cp_copy.JUMP1.select_toa_mask(setup_NGC6440E.t),
        )
    )


if __name__ == "__main__":
    pass
