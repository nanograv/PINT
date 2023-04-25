"""Test model builder using variance input"""

import pytest
import io
from glob import glob
import copy
from os.path import basename, join
import numpy as np
from pint.models.timing_model import (
    TimingModel,
    PhaseComponent,
    Component,
    AllComponents,
    AliasConflict,
)
from pint.models.model_builder import ModelBuilder, ComponentConflict, get_model
from pint.models.parameter import floatParameter
from pint.utils import split_prefixed_name, PrefixError
from pinttestdata import datadir


class SimpleModel(PhaseComponent):
    """Very simple test model component"""

    register = True
    category = "simple_test"

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.add_param(floatParameter(name="TESTPARAM", value=0.0, unit="s"))


class SubsetModel(PhaseComponent):
    """Test model component hosting the parameters which are a subset of spindown."""

    register = False  # This has to be false, otherwrise all test will fail.
    category = "simple_test"

    def __init__(self):
        super(SubsetModel, self).__init__()
        self.add_param(floatParameter(name="F0", value=0.0, unit="1/s"))
        # self.add_param(floatParameter(name="F1", value=0.0, unit="1/s^2"))


@pytest.fixture
def sub_set_model():
    return SubsetModel()


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def simple_model_overlap(simple_model):
    simple_model.add_param(floatParameter(name="F0", aliases=[], value=0.0, unit="1/s"))
    return simple_model


@pytest.fixture
def simple_model_alias_overlap():
    simple_model = SimpleModel()
    simple_model.add_param(
        floatParameter(name="TESTPARAM2", aliases=["F0"], value=0.0, unit="1/s")
    )
    simple_model.add_param(
        floatParameter(name="TESTPARAM3", aliases=["LAMBDA"], value=0.0, unit="deg")
    )
    return simple_model


@pytest.fixture
def test_timing_model():
    ac = AllComponents()
    return TimingModel(
        name="Test",
        components=[
            ac.components["AstrometryEquatorial"],
            ac.components["Spindown"],
            ac.components["DispersionDMX"],
            ac.components["PhaseJump"],
            ac.components["ScaleToaError"],
        ],
    )


pint_dict_base = {
    "PSR": ["J1234+5678"],
    "RAJ": ["04:37:15.7865145       1   7.000e-07"],
    "DECJ": ["-47:15:08.461584       1   8.000e-06"],
    "F0": ["173.6879489990983      1   3.000e-13"],
    "PEPOCH": ["51194.000"],
    "JUMP1": ["-fe L-wide 1 1 0.1", "-fe 430 1 1 0.1", "-fe1 430 1 1 0.1"],
    "DMX_0001": ["3.01718358D-03  1      3.89019948D-05"],
    "EFAC1": ["-f L-wide_PUPPI   1.156", "-f 430_ASP   0.969"],
    "EQUAD1": ["-f L-wide_PUPPI   0.14320"],
}


def test_model_builder_class():
    """Test if AllComponents collected components information correctly"""
    mb = AllComponents()
    category = mb.category_component_map
    assert set(mb.param_component_map["PX"]) == set(category["astrometry"])
    assert set(mb.component_category_map.keys()) == set(mb.components)
    # test for new components
    assert "SimpleModel" in mb.components
    simple_comp = mb.components["SimpleModel"]
    simple_comp.add_param(
        floatParameter(name="TESTPARAM2", aliases=["F0"], value=0.0, unit="s")
    )


def test_aliases_mapping():
    """Test if aliases gets mapped correctly"""
    mb = AllComponents()
    # all aliases should be mapped to the components
    assert set(mb._param_alias_map.keys()) == set(mb.param_component_map.keys())

    # Test if the param_alias_map is passed by pointer
    # Testing the private function for building the aliases map
    mb._check_alias_conflict("TESTAX", "TESTAXX", mb._param_alias_map)
    # assert "TESTAX" in mb._param_alias_map
    # Test existing entry
    # When adding an existing alias to the map. The mapped value should be the
    # same, otherwise it will fail.
    mb._check_alias_conflict("F0", "F0", mb._param_alias_map)
    # assert mb._param_alias_map["F0"] == "F0"
    # Test repeatable_params with different indices.
    for rp in mb.repeatable_param:
        pint_par, first_init_par = mb.alias_to_pint_param(rp)
        cp = mb.param_component_map[pint_par][0]
        pint_par_obj = getattr(mb.components[cp], pint_par)
        try:
            prefix, id, ids = split_prefixed_name(rp)
        except PrefixError:
            prefix = rp

        new_idx_par = f"{prefix}2"
        assert mb.alias_to_pint_param(new_idx_par)[0] == f"{pint_par_obj.prefix}2"
        new_idx_par = f"{prefix}55"
        assert mb.alias_to_pint_param(new_idx_par)[0] == f"{pint_par_obj.prefix}55"
        # Test all aliases
        for als in pint_par_obj.aliases:
            assert mb.alias_to_pint_param(als)[0] == pint_par_obj.name
            try:
                als_prefix, id, ids = split_prefixed_name(als)
            except PrefixError:
                als_prefix = als
        assert mb.alias_to_pint_param(f"{als_prefix}2")[0] == f"{pint_par_obj.prefix}2"
        assert (
            mb.alias_to_pint_param(f"{als_prefix}55")[0] == f"{pint_par_obj.prefix}55"
        )


def test_conflict_alias():
    """Test if model builder detects the alias conflict."""
    mb = AllComponents()
    # Test conflict parameter alias name
    with pytest.raises(AliasConflict):
        mb._check_alias_conflict("F0", "F1", mb._param_alias_map)


def test_conflict_alias_in_component():
    # Define conflict alias from component class
    class SimpleModel2(PhaseComponent):
        """Very simple test model component"""

        register = True
        category = "simple_test"

        def __init__(self):
            super(SimpleModel2, self).__init__()
            self.add_param(
                floatParameter(name="TESTPARAMF0", aliases=["F0"], value=0.0, unit="s")
            )

    mb2 = AllComponents()
    with pytest.raises(AliasConflict):
        mb2._param_alias_map
    del Component.component_types["SimpleModel2"]


def test_overlap_component(simple_model_overlap, simple_model_alias_overlap):
    """Test if model builder detects the overlap component correctly."""
    mb = ModelBuilder()
    # Test overlap
    overlap = mb._get_component_param_overlap(simple_model_overlap)
    assert "Spindown" in overlap.keys()
    assert overlap["Spindown"][0] == {"F0"}
    # Only one over lap parameter F0
    # Since the _get_component_param_overlap returns non-overlap part,
    # we test if the non-overlap number makes sense.
    assert overlap["Spindown"][1] == len(simple_model_overlap.params) - 1
    assert (
        overlap["Spindown"][2]
        == len(mb.all_components.components["Spindown"].params) - 1
    )

    a_overlap = mb._get_component_param_overlap(simple_model_alias_overlap)
    assert a_overlap["Spindown"][0] == {"F0"}
    assert a_overlap["Spindown"][1] == len(simple_model_alias_overlap.params) - 1
    assert (
        a_overlap["Spindown"][2]
        == len(mb.all_components.components["Spindown"].params) - 1
    )
    assert a_overlap["AstrometryEcliptic"][0] == {"ELONG"}
    assert (
        a_overlap["AstrometryEcliptic"][1] == len(simple_model_alias_overlap.params) - 1
    )
    assert (
        a_overlap["AstrometryEcliptic"][2]
        == len(mb.all_components.components["AstrometryEcliptic"].params) - 1
    )


def test_subset_component(sub_set_model):
    """Test if model builder detects the subset component."""
    mb = ModelBuilder()
    # Test subset
    superset = mb._is_subset_component(sub_set_model)
    assert superset == "Spindown"


def test_model_fillup(test_timing_model):
    """Test model value fill up"""
    mb = ModelBuilder()
    tm = mb._setup_model(test_timing_model, pint_dict_base, validate=False)
    assert tm.PSR.value == "J1234+5678"
    assert np.isclose(tm.F0.value, 173.6879489990983)
    assert np.isclose(tm.F0.uncertainty_value, 3.000e-13)
    assert tm.DMX_0001.value == 3.01718358e-03
    assert tm.DMX_0001.uncertainty_value == 3.89019948e-05
    jump_map = tm.get_prefix_mapping("JUMP")
    assert len(jump_map.keys()) == len(pint_dict_base["JUMP1"])
    assert tm.JUMP1.key == "-fe"
    assert tm.JUMP1.key_value == ["L-wide"]
    assert tm.JUMP1.value == 1
    assert tm.JUMP2.key == "-fe"
    assert tm.JUMP2.key_value == ["430"]
    assert tm.JUMP2.value == 1
    assert tm.JUMP3.key == "-fe1"
    assert tm.JUMP3.key_value == ["430"]
    assert tm.JUMP3.value == 1
    efac = tm.get_prefix_mapping("EFAC")
    assert len(efac.keys()) == len(pint_dict_base["EFAC1"])
    assert tm.EFAC1.key == "-f"
    assert tm.EFAC1.key_value == ["L-wide_PUPPI"]
    assert tm.EFAC1.value == 1.156
    assert tm.EFAC2.key == "-f"
    assert tm.EFAC2.key_value == ["430_ASP"]
    assert tm.EFAC2.value == 0.969
    equad = tm.get_prefix_mapping("EQUAD")
    assert len(equad.keys()) == len(pint_dict_base["EQUAD1"])


def test_model_fillup_prefix_adding(test_timing_model):
    pint_dict_prefix = copy.deepcopy(pint_dict_base)
    pint_dict_prefix["DMX_0002"] = ["5.01718358D-03  1      3.89019948D-05"]
    pint_dict_prefix["DMX_0345"] = ["3.01718358D-03  1      5.89019948D-05"]
    mb = ModelBuilder()
    tm = mb._setup_model(test_timing_model, pint_dict_prefix, validate=False)
    assert np.isclose(tm.DMX_0002.value, 5.01718358e-03)
    assert np.isclose(tm.DMX_0002.uncertainty_value, 3.89019948e-05)
    assert np.isclose(tm.DMX_0345.value, 3.01718358e-03)
    assert np.isclose(tm.DMX_0345.uncertainty_value, 5.89019948e-05)


def test_model_fillup_prefix_adding_spin_freq(test_timing_model):
    pint_dict_prefix = copy.deepcopy(pint_dict_base)
    pint_dict_prefix["F1"] = ["5.0D-11  1      3.0e-15"]
    pint_dict_prefix["F2"] = ["5.0D-13  1      3.0e-15"]
    pint_dict_prefix["F3"] = ["3.0D-14  1      5.0e-15"]
    mb = ModelBuilder()
    tm = mb._setup_model(test_timing_model, pint_dict_prefix)
    assert np.isclose(tm.F2.value, 5.0e-13)
    assert np.isclose(tm.F2.uncertainty_value, 3.0e-15)
    assert np.isclose(tm.F3.value, 3.0e-14)
    assert np.isclose(tm.F3.uncertainty_value, 5.0e-15)


def test_model_from_par():
    """Test Get model from test par file."""
    test_par1 = """
    PSR              B1855+09
    LAMBDA   286.8634893301156  1     0.0000000165859
    BETA      32.3214877555037  1     0.0000000273526
    PMLAMBDA           -3.2701  1              0.0141
    PMBETA             -5.0982  1              0.0291
    PX                  0.2929  1              0.2186
    F0    186.4940812707752116  1  0.0000000000328468
    F1     -6.205147513395D-16  1  1.379566413719D-19
    PEPOCH        54978.000000
    START            53358.726
    FINISH           56598.873
    DM               13.299393
    DMX              14.000000
    DMX_0001    1.51618630D-02  1      3.51684846D-03
    DMXEP_0001     53358.72746
    DMXR1_0001     53358.72746
    DMXR2_0001     53358.77841
    DMXF1_0001         424.000
    DMXF2_0001        1442.000
    DMX_0002    1.52370685D-02  1      3.51683449D-03
    DMXEP_0002     53420.54893
    DMXR1_0002     53420.54893
    DMXR2_0002     53420.58620
    DMXF1_0002         424.000
    DMXF2_0002        1442.000
    FD1  1.61666384D-04  1      3.38650356D-05
    FD2 -1.88210030D-04  1      4.13173074D-05
    SOLARN0               0.00
    EPHEM               DE421
    ECL                 IERS2003
    CLK                 TT(BIPM)
    UNITS               TDB
    TIMEEPH             FB90
    T2CMETHOD           TEMPO
    CORRECT_TROPOSPHERE N
    PLANET_SHAPIRO      N
    #DILATEFREQ          N
    NTOA                  4005
    TRES                  5.52
    TZRMJD  54981.28084616488447
    TZRFRQ             424.000
    TZRSITE                  3
    NITS                     1
    INFO -f
    BINARY            DD
    A1             9.230780480  1         0.000000203
    E             0.0000216340  1        0.0000000236
    T0        54975.5128660817  1        0.0019286695
    PB       12.32717119132762  1    0.00000000019722
    OM        276.536118059963  1      0.056323656112
    SINI              0.999461  1            0.000178
    M2                0.233837  1            0.011278
    RNAMP         0.17173D-01
    RNIDX            -4.91353
    TNRedAmp -14.227505410948254
    TNRedGam 4.91353
    TNRedC 45
    T2EFAC -f L-wide_PUPPI   1.507
    T2EFAC -f 430_ASP   1.147
    T2EQUAD -f L-wide_PUPPI   0.25518
    T2EQUAD -f 430_ASP   0.01410
    ECORR -f 430_PUPPI   0.00601
    ECORR -f L-wide_PUPPI   0.31843
    JUMP -fe L-wide      -0.000009449  1       0.000009439
    """
    mb = ModelBuilder()
    param_inpar, original_name, unknown = mb._pintify_parfile(io.StringIO(test_par1))
    # valid_param_inline = []
    # for l in test_par1.split("\n"):
    #     l = l.strip()
    #     if not (l.startswith("#") or l == ""):
    #         valid_param_inline.append(l.split()[0])
    # assert set(repeat) == {"JUMP", "ECORR", "T2EQUAD", "T2EFAC"}
    comps, conflict, _ = mb.choose_model(param_inpar)
    assert comps == {
        "Spindown",
        "FD",
        "AbsPhase",
        "ScaleToaError",
        "TroposphereDelay",
        "PhaseJump",
        "DispersionDMX",
        "AstrometryEcliptic",
        "BinaryDD",
        "DispersionDM",
        "EcorrNoise",
        "SolarWindDispersion",
        "PLRedNoise",
        "SolarSystemShapiro",
    }
    tm = mb(io.StringIO(test_par1))
    assert len(tm.get_prefix_mapping("EQUAD")) == 2
    assert len(tm.get_prefix_mapping("EFAC")) == 2
    assert len(tm.get_prefix_mapping("JUMP")) == 1


def test_model_from_par_hassubset():
    """Test Get model from test par file with a subset component."""

    # Define a subset parameter model that is registered. So the metaclass can
    # catch it.
    class SubsetModel2(PhaseComponent):
        """Test model component hosting the parameters which are a subset of spindown."""

        register = True
        category = "simple_test"

        def __init__(self):
            super(SubsetModel2, self).__init__()
            self.add_param(floatParameter(name="F0", value=0.0, unit="1/s"))

    with pytest.raises(ComponentConflict):
        mb = ModelBuilder()
    # Have to remove the SubsetModel2, since it will fail other tests.
    del Component.component_types["SubsetModel2"]


bad_trouble = ["J1923+2515_NANOGrav_9yv1.gls.par", "J1744-1134.basic.ecliptic.par"]


# Test all the parameters.
@pytest.mark.parametrize("parfile", glob(join(datadir, "*.par")))
def test_all_parfiles(parfile):
    if basename(parfile) in bad_trouble:
        pytest.skip("This parfile is unclear")
    model = get_model(parfile)


def test_include_solar_system_shapiro():
    par = "F0 100 1"
    m = get_model(io.StringIO(par))
    assert "SolarSystemShapiro" not in m.components

    par = """
        ELAT 0.1 1
        ELONG 2.1 1
        F0 100 1 
    """
    m = get_model(io.StringIO(par))
    assert "SolarSystemShapiro" in m.components

    par = """
        RAJ 06:00:00 1
        DECJ 12:00:00 1
        F0 100 1 
    """
    m = get_model(io.StringIO(par))
    assert "SolarSystemShapiro" in m.components
