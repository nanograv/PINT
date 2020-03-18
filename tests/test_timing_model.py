"""
"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
import astropy.units as u
from pint.models import TimingModel, DEFAULT_ORDER
from pint.models import get_model
from pint.models import (
    AstrometryEquatorial,
    Spindown,
    DelayJump,
    DispersionDM,
    BinaryELL1,
    Wave,
)
from pint.models import parameter as p
from pinttestdata import datadir


class TestModelBuilding:
    def setup(self):
        self.parfile = os.path.join(datadir, "J0437-4715.par")

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
        tm = TimingModel(
            "TestTimingModel",
            [BinaryELL1(), Wave(), AstrometryEquatorial(), Spindown()],
        )

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
        tm = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )

        tm.add_component(DelayJump(), validate=False)
        # Test link
        # TODO may be add a get_component function
        cp = tm.components["DelayJump"]
        assert cp._parent == tm

        # Test order
        cp_pos = tm.DelayComponent_list.index(cp)
        assert cp_pos == 2

        print(cp.params)
        print(cp.get_prefix_mapping_component("JUMP"))
        print(id(cp), "test")
        add_jumps = [
            ("JUMP", {"value": 0.1, "key": "mjd", "key_value": [55000, 56000]}),
            ("JUMP", {"value": 0.2, "key": "freq", "key_value": [1440, 2000]}),
            ("JUMP", {"value": 0.3, "key": "tel", "key_value": "ao"}),
        ]

        for jp in add_jumps:
            p_name = jp[0]
            print("test1", p_name)
            p_vals = jp[1]
            par = p.maskParameter(
                name=p_name,
                key=p_vals["key"],
                value=p_vals["value"],
                key_value=p_vals["key_value"],
                units=u.s,
            )
            print("test", par.name)
            cp.add_param(par, setup=True)
        # TODO add test component setup function. use jump1 right now, this
        # should be updated in the future.
        assert hasattr(cp, "JUMP1")
        assert hasattr(cp, "JUMP2")
        assert hasattr(cp, "JUMP3")
        assert hasattr(tm, "JUMP1")
        assert hasattr(tm, "JUMP2")
        assert hasattr(tm, "JUMP3")
        jump1 = getattr(tm, "JUMP1")
        jump2 = getattr(tm, "JUMP2")
        jump3 = getattr(tm, "JUMP3")
        assert jump1.key == "mjd"
        assert jump2.key == "freq"
        assert jump3.key == "tel"
        # Check jump value
        assert jump1.value == 0.1
        assert jump2.value == 0.2
        assert jump3.value == 0.3
        # Check jump key value
        assert jump1.key_value == [55000, 56000]
        assert jump2.key_value == [1440, 2000]
        assert jump3.key_value == ["ao"]
        assert tm.jumps == ["JUMP1", "JUMP2", "JUMP3"]

    def test_remove_component(self):
        tm = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )

        remove_cp = tm.components["BinaryELL1"]

        # test remove by name
        tm.remove_component("BinaryELL1")
        assert not "BinaryELL1" in tm.components.keys()
        assert not remove_cp in tm.DelayComponent_list

        # test remove by component
        tm2 = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )

        remove_cp2 = tm2.components["BinaryELL1"]
        tm2.remove_component(remove_cp2)
        assert not "BinaryELL1" in tm2.components.keys()
        assert not remove_cp2 in tm2.DelayComponent_list
