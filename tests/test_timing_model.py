"""
"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy
from pint.models import TimingModel
from pint.models.timing_model import DEFAULTORDER
from pint.models import get_model
from pint.models import (
    AstrometryEquatorial,
    Spindown,
    DelayJump,
    DispersionDM,
    BinaryELL1,
    Wave
)
from pinttestdata import datadir


class TestModelBuilding:

    def setup(self):
        self.parfile = os.path.join(datadir,
                                    "J0437-4715.par")

    def test_from_par(self):
        tm = get_model(self.parfile)
        assert len(tm.componenets) == 6
        assert len(tm.DelayComponent_list) == 4
        assert len(tm.PhaseComponent_list) == 2

        # Check delay component order
        order = []
        for dcp in tm.DelayComponent_list:
            order.append(DEFAULTORDER.index(dcp.category))
        assert np.diff(np.array(order)) > 0
        # Check phase component order
        order = []
        for dcp in tm.PhaseComponent_list:
            order.append(DEFAULTORDER.index(dcp.category))
        assert np.diff(np.array(order)) > 0

    def test_from_scratch(self):
        tm = TimingModel([BinaryELL1(), Wave(), AstrometryEquatorial(),
                          Spindown()])

        for k, v in tm.components.items():
            # test the link to timing model
            assert v._parent == tm

        # Test Delay order
        order = []
        for dcp in tm.DelayComponent_list:
            order.append(DEFAULTORDER.index(dcp.category))
        assert np.diff(np.array(order)) > 0

        # Test Phase order
        order = []
        for dcp in tm.PhaseComponent_list:
            order.append(DEFAULTORDER.index(dcp.category))
        assert np.diff(np.array(order)) > 0

    def test_add_component(self):
        tm = TimingModel([BinaryELL1(), AstrometryEquatorial(),
                          Spindown()])

        tm.add_component(DelayJump())
        # Test link
        # TODO may be add a get_component function
        cp = tm.components('DelayJump')
        assert cp._parent == tm

        # Test order
        cp_pos = tm.DelayComponent_list.index(cp)
        assert cp_pos == 2

        # TODO add test component setup function.
