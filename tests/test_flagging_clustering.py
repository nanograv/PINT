"""Tests for clustering and flagging"""
import os
import pytest
import copy

import astropy.units as u

import pint.models.model_builder as mb
import pint.toa as toa
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
    return SimpleSetup(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )


def test_jump_by_cluster(setup_NGC6440E):
    """
    Compare selection by MJD with selection by cluster (which by default selects TOAs by 2h gaps)
    """
    m_copy = copy.deepcopy(setup_NGC6440E.m)

    setup_NGC6440E.m.add_component(PhaseJump(), validate=False)
    cp = setup_NGC6440E.m.components["PhaseJump"]
    par = p.maskParameter(
        name="JUMP", key="mjd", value=0.2, key_value=[54099, 54100], units=u.s
    )
    # this should be the last group of any TOAs
    cp.add_param(par, setup=True)

    # add clusters to the TOAs
    clusters = setup_NGC6440E.t.get_clusters(
        2 * u.hr, add_column=False, add_flag="cluster"
    )

    m_copy.add_component(PhaseJump(), validate=False)
    cp_copy = m_copy.components["PhaseJump"]
    par_copy = p.maskParameter(
        name="JUMP", key="-cluster", value=0.2, key_value=41, units=u.s
    )
    # this should be identical to the cluster above
    cp_copy.add_param(par_copy, setup=True)
    assert set(cp.JUMP1.select_toa_mask(setup_NGC6440E.t)) == set(
        cp_copy.JUMP1.select_toa_mask(setup_NGC6440E.t)
    )


def test_jump_by_cluster_invalidflags(setup_NGC6440E):
    # add clusters to the TOAs
    t = copy.deepcopy(setup_NGC6440E.t)
    with pytest.raises(ValueError):
        clusters = t.get_clusters(
            2 * u.hr,
            add_column=False,
            add_flag=True,
        )

    # add clusters to the TOAs
    t = copy.deepcopy(setup_NGC6440E.t)
    with pytest.raises(ValueError):
        clusters = t.get_clusters(
            2 * u.hr,
            add_column=False,
            add_flag=1,
        )
