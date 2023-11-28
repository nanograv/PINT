import io
import os
import sys
import warnings
from copy import deepcopy
from contextlib import redirect_stdout

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pinttestdata import datadir

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
from pint.simulation import make_fake_toas_uniform
from pint.toa import get_TOAs


@pytest.fixture
def model_0437():
    return get_model(os.path.join(datadir, "J0437-4715.par"))


@pytest.fixture
def timfile_jumps():
    os.chdir(datadir)
    return get_TOAs("test1.tim")


@pytest.fixture
def timfile_nojumps():
    return get_TOAs(os.path.join(datadir, "NGC6440E.tim"))


len_timfile_nojumps = len(get_TOAs(os.path.join(datadir, "NGC6440E.tim")))


class TestModelBuilding:
    def setup_method(self):
        self.parfile = os.path.join(datadir, "J0437-4715.par")

    def test_from_par(self):
        tm = get_model(self.parfile)
        assert tm.UNITS.value == "TDB"
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
        tfp = {"F0", "TASC", "EPS1", "RAJ"}
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
            tm.free_params = ["F0", "TASC", "EPS1", "RAJ", "CAPYBARA"]

        tfp = {"F0", "TASC", "EPS1", "RAJ"}
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
    make_fake_toas_uniform(56000, 57000, 10, model=model)


def test_t2cmethod_corrected():
    with pytest.warns(UserWarning, match=".*T2CMETHOD.*"):
        model = get_model(io.StringIO("\n".join([par_base, "T2CMETHOD TEMPO"])))
    assert model.T2CMETHOD.value == "IAU2000B"


def test_jump_flags_to_params_harmless(timfile_nojumps, model_0437):
    # TOAs 9, 10, 11, and 12 have jump flags (JUMP2 on 9, JUMP1 on rest)
    m = model_0437  # model with no jumps
    t_nojump = timfile_nojumps
    # sanity check
    assert "PhaseJump" not in m.components
    # check nothing changed when .tim file has no jumps
    m.jump_flags_to_params(t_nojump)
    assert "PhaseJump" not in m.components


def test_jump_flags_to_params_adds_params(timfile_jumps, model_0437):
    # TOAs 9, 10, 11, and 12 have jump flags (JUMP2 on 9, JUMP1 on rest)
    t = timfile_jumps
    m = model_0437  # model with no jumps

    m.jump_flags_to_params(t)
    assert "PhaseJump" in m.components
    assert len(m.components["PhaseJump"].jumps) == 2
    assert "JUMP1" in m.components["PhaseJump"].jumps
    assert "JUMP2" in m.components["PhaseJump"].jumps


def test_jump_flags_to_params_idempotent(timfile_jumps, model_0437):
    # TOAs 9, 10, 11, and 12 have jump flags (JUMP2 on 9, JUMP1 on rest)
    t = timfile_jumps
    m = model_0437  # model with no jumps

    m.jump_flags_to_params(t)
    m.jump_flags_to_params(t)
    assert "PhaseJump" in m.components
    assert len(m.components["PhaseJump"].jumps) == 2
    assert "JUMP1" in m.components["PhaseJump"].jumps
    assert "JUMP2" in m.components["PhaseJump"].jumps


def test_many_timfile_jumps():
    m = get_model(io.StringIO(par_base))
    pairs = 15
    toas_per_jump = 3
    t = make_fake_toas_uniform(56000, 57000, 5 + toas_per_jump * pairs, model=m)
    # The following lets us write the fake TOAs to a string as a timfile
    f = io.StringIO()
    with redirect_stdout(f):
        t.write_TOA_file(sys.stdout)
    s = f.getvalue().splitlines()
    toalist = ["\n".join(s[:11]) + "\n"]
    lo = 12
    for _ in range(pairs):
        toalist.append("\n".join(["JUMP"] + s[lo : lo + toas_per_jump] + ["JUMP\n"]))
        lo += toas_per_jump
    # read the TOAs
    tt = get_TOAs(io.StringIO("".join(toalist)))
    # convert the timfile JUMPs to params
    m.jump_flags_to_params(tt)
    assert "PhaseJump" in m.components
    assert len(m.components["PhaseJump"].jumps) == pairs
    assert "JUMP1" in m.components["PhaseJump"].jumps
    assert "JUMP2" in m.components["PhaseJump"].jumps
    assert "JUMP10" in m.components["PhaseJump"].jumps
    assert "JUMP15" in m.components["PhaseJump"].jumps


def test_parfile_and_timfile_jumps(timfile_jumps):
    t = timfile_jumps
    # this test assumes things have been grouped by observatory to associate the jumps with specific TOA indices
    t.table = t.table.group_by("obs")
    fs_orig, idxs_orig = t.get_flag_value("tim_jump")
    m = get_model(io.StringIO(par_base + "JUMP MJD 55729 55730 0.0 1\n"))
    # turns pre-existing jump flags in t.table['flags'] into parameters in parfile
    m.jump_flags_to_params(t)

    assert "PhaseJump" in m.components
    fs, idxs = t.get_flag_value("tim_jump")
    assert fs == fs_orig
    assert idxs == idxs_orig

    # adds jump flags to t.table['flags'] for jump parameters already in parfile
    m.jump_params_to_flags(t)
    fs, idxs = t.get_flag_value("jump")
    assert len(idxs) == 5
    assert fs[5] in ["3,1", "1,3"]
    assert fs[4] == "1"


def test_supports_rm():
    m = get_model(io.StringIO("\n".join([par_base, "RM 10"])))
    assert m.RM.value == 10


def test_assumes_dmepoch_equals_pepoch():
    m_assume = get_model(io.StringIO("\n".join([par_base, "DM1 1e-5"])))
    m_given = get_model(io.StringIO("\n".join([par_base, "DMEPOCH 58000", "DM1 1e-5"])))

    t = make_fake_toas_uniform(57000, 59000, 10, m_assume)

    assert_allclose(m_assume.dm_value(t), m_given.dm_value(t))


def test_prefixed_aliases_in_component():
    m = get_model(
        io.StringIO("\n".join([par_base, "T2EFAC -fe L_band 10", "T2EFAC -fe 430 11"]))
    )
    assert m.components["ScaleToaError"].aliases_map["T2EFAC2"] == "EFAC2"
    with pytest.raises(KeyError):
        m.components["ScaleToaError"].aliases_map["T2EFAC18"]


def test_writing_equals_string(model_0437):
    f = io.StringIO()
    model_0437.write_parfile(f, include_info=False)
    assert f.getvalue() == model_0437.as_parfile(include_info=False)


def test_writing_to_file_equals_string(tmp_path, model_0437):
    p = tmp_path / "file.par"
    model_0437.write_parfile(p, include_info=False)
    assert p.read_text() == model_0437.as_parfile(include_info=False)


def test_dispersion_slope(model_0437):
    toas = make_fake_toas_uniform(56000, 57000, 50, model=model_0437)
    dsl = model_0437.total_dispersion_slope(toas)
    assert np.all(np.isfinite(dsl))
