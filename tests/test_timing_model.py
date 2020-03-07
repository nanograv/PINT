"""
"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
from pint.models import TimingModel, DEFAULT_ORDER
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
        assert all(np.diff(np.array(order)) > 0)
        # Check phase component order
        order = []
        for dcp in tm.PhaseComponent_list:
            order.append(DEFAULT_ORDER.index(dcp.category))
        assert all(np.diff(np.array(order)) > 0)

    def test_component_input(self):
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
            order.append(DEFAULT_ORDER.index(dcp.category))
        assert all(np.diff(np.array(order)) > 0)

        # Test Phase order
        order = []
        for dcp in tm.PhaseComponent_list:
            order.append(DEFAULT_ORDER.index(dcp.category))
        assert all(np.diff(np.array(order)) > 0)

    def test_add_component(self):
        tm = TimingModel("TestTimingModel", [BinaryELL1(),
                                             AstrometryEquatorial(),
                                             Spindown()])

        tm.add_component(DelayJump(), [('JUMP', {'value': 0.1, 'key': 'mjd',
                                                'key_value': [55000, 56000]}),
                                       ('JUMP', {'value': 0.2, 'key': 'freq',
                                                'key_value': [1440, 2000]}),
                                       ('JUMP', {'value': 0.3, 'key': 'tel',
                                                'key_value': 'ao'})],
                                        build_mood=False)
        # Test link
        # TODO may be add a get_component function
        cp = tm.components['DelayJump']
        assert cp._parent == tm

        # Test order
        cp_pos = tm.DelayComponent_list.index(cp)
        assert cp_pos == 2

        # TODO add test component setup function. use jump1 right now, this
        # should be updated in the future.
        assert hasattr(cp, 'JUMP1')
        assert hasattr(cp, 'JUMP2')
        assert hasattr(cp, 'JUMP3')
        assert hasattr(tm, 'JUMP1')
        assert hasattr(tm, 'JUMP2')
        assert hasattr(tm, 'JUMP3')
        jump1 = getattr(tm, 'JUMP1')
        jump2 = getattr(tm, 'JUMP2')
        jump3 = getattr(tm, 'JUMP3')
        assert jump1.key == 'mjd'
        assert jump2.key == 'freq'
        assert jump3.key == 'tel'
        # Check jump value
        assert jump1.value == 0.1
        assert jump2.value == 0.2
        assert jump3.value == 0.3
        # Check jump key value
        assert jump1.key_value == [55000, 56000]
        assert jump2.key_value == [1440, 2000]
        assert jump3.key_value == 'ao'
        assert tm.jumps == ['JUMP1', 'JUMP2', 'JUMP3']

    def test_add_param(self):
        tm1 = TimingModel("TestTimingModel", [BinaryELL1(),
                                             AstrometryEquatorial(),
                                             DispersionDMX(),
                                             DelayJump(),
                                             Spindown()])
        # Add parameters
