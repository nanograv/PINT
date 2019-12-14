"""
"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
from pint.models import TimingModel
from pint.models.timing_model import DEFAULT_ORDER
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
        assert len(tm.components) == 6
        assert len(tm.DelayComponent_list) == 4
        assert len(tm.PhaseComponent_list) == 2

        # Check delay component order
        order = []
        for dcp in tm.DelayComponent_list:
            order.append(DEFAULT_ORDER.index(dcp.category))
        assert np.diff(np.array(order)) > 0
        # Check phase component order
        order = []
        for dcp in tm.PhaseComponent_list:
            order.append(DEFAULT_ORDER.index(dcp.category))
        assert np.diff(np.array(order)) > 0

    def test_from_scratch(self):
        tm = TimingModel("TestTimingModel", [BinaryELL1(),
                                             Wave(),
                                             AstrometryEquatorial(),
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
        tm = TimingModel("TestTimingModel", [BinaryELL1(),
                                             AstrometryEquatorial(),
                                             Spindown()])

        tm.add_component(DelayJump(), {'JUMP': {'value': 0, 'key': 'mjd',
                                                'key_value': [55000, 56000]}})
        # Test link
        # TODO may be add a get_component function
        cp = tm.components('DelayJump')
        assert cp._parent == tm

        # Test order
        cp_pos = tm.DelayComponent_list.index(cp)
        assert cp_pos == 2

        # TODO add test component setup function.
        assert hasattr(cp, 'JUMP')
