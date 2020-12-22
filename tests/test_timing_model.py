from __future__ import absolute_import, division, print_function

import io
import os
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir
from pint.toa import make_fake_toas

from pint.models import (
    DEFAULT_ORDER,
    AstrometryEquatorial,
    BinaryELL1,
    DelayJump,
    Spindown,
    TimingModel,
    Wave,
    get_model,
    parameter as p,
)


@pytest.fixture
def model_0437():
    return get_model(os.path.join(datadir, "J0437-4715.par"))


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
        assert jump2.key_value == [1440 * u.MHz, 2000 * u.MHz]
        assert jump3.key_value == ["arecibo"]
        assert tm.jumps == ["JUMP1", "JUMP2", "JUMP3"]

    def test_remove_component(self):
        tm = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )

        remove_cp = tm.components["BinaryELL1"]

        # test remove by name
        tm.remove_component("BinaryELL1")
        assert "BinaryELL1" not in tm.components.keys()
        assert remove_cp not in tm.DelayComponent_list

        # test remove by component
        tm2 = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )

        remove_cp2 = tm2.components["BinaryELL1"]
        tm2.remove_component(remove_cp2)
        assert "BinaryELL1" not in tm2.components.keys()
        assert remove_cp2 not in tm2.DelayComponent_list

    def test_free_params(self):
        """Test getting free parameters."""
        # Build the timing model
        tm = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )
        tfp = {"F0", "T0", "EPS1", "RAJ"}
        # Turn off the fit parameters
        for p in tm.params:
            par = getattr(tm, p)
            par.frozen = p not in tfp
        assert set(tm.free_params) == tfp

    def test_change_free_params(self):
        """Test setting free parameters."""
        # Build the timing model
        tm = TimingModel(
            "TestTimingModel", [BinaryELL1(), AstrometryEquatorial(), Spindown()]
        )

        with pytest.raises(ValueError):
            tm.free_params = ["F0", "T0", "EPS1", "RAJ", "CAPYBARA"]

        tfp = {"F0", "T0", "EPS1", "RAJ"}
        tm.free_params = tfp
        assert set(tm.free_params) == tfp

    def test_params_dict_round_trip_quantity(self):
        tm = get_model(self.parfile)
        tfp = {"F0", "T0", "RAJ"}
        tm.free_params = tfp
        tm.set_param_values(tm.get_params_dict("free", "quantity"))

    def test_params_dict_round_trip_num(self):
        tm = get_model(self.parfile)
        tfp = {"F0", "T0", "RAJ"}
        tm.free_params = tfp
        tm.set_param_values(tm.get_params_dict("free", "num"))

    def test_params_dict_round_trip_uncertainty(self):
        tm = get_model(self.parfile)
        tfp = {"F0", "T0", "RAJ"}
        tm.free_params = tfp
        tm.set_param_uncertainties(tm.get_params_dict("free", "uncertainty"))


def test_parameter_access(model_0437):
    model_0437.F0


def test_component_parameter_access(model_0437):
    model_0437.components["Spindown"].F0


def test_component_methods(model_0437):
    model_0437.coords_as_GAL


def test_copying(model_0437):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deepcopy(model_0437)


def test_in(model_0437):
    assert "F0" in model_0437
    assert "CAPYBARA" not in model_0437


def test_getitem(model_0437):
    assert model_0437["F0"] is model_0437.F0
    with pytest.raises(KeyError):
        model_0437["CAPYBARA"]


def test_listify(model_0437):
    assert list(model_0437) == model_0437.params


def test_len(model_0437):
    assert len(model_0437) == len(model_0437.params)


def test_keys(model_0437):
    assert model_0437.keys() == model_0437.params


def test_items(model_0437):
    its = model_0437.items()
    assert len(its) == len(model_0437)
    for po, (pi, v) in zip(model_0437.params, its):
        assert po == pi
        assert model_0437[po] == v


def test_iterator(model_0437):
    assert [k for k in model_0437] == model_0437.params


par_base = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 58000
F0 1
DM 10
"""


@pytest.mark.parametrize(
    "lines,param,value",
    [
        ([], "DMJUMP", 1),
        (["DMJUMP -fe L_band 10"], "DMJUMP", 1),
        ([], "H0", 1),
        ([], "F2", 1),
    ],
)
def test_set_params(lines, param, value):
    model = get_model(io.StringIO("\n".join([par_base] + lines)))
    with pytest.raises(KeyError) as e:
        model[param].value = value
    str(e.value).index(param)  # raise exception if not found
    with pytest.raises(AttributeError) as e:
        getattr(model, param).value = value
    str(e.value).index(param)  # raise exception if not found


@pytest.mark.parametrize(
    "lines,param,exception",
    [
        ([], "garbage_parameter", ValueError),
        ([], "H4", ValueError),
        (["DMJUMP -fe L_band 10", "DMJUMP -fe S_band 20"], "DMJUMP", ValueError),
    ],
)
def test_free_params(lines, param, exception):
    model = get_model(io.StringIO("\n".join([par_base] + lines)))
    with pytest.raises(exception):
        model.free_params = [param]


def test_pepoch_late():
    model = get_model(io.StringIO(par_base))
    t = make_fake_toas(56000, 57000, 10, model=model)
