"""Test model builder using variance of parfile input"""

from collections import defaultdict
import pytest
import io
import astropy.units as u
from pint.models.model_builder import (
    ModelBuilder,
    ConflictAliasError,
    UnknownBinaryModel,
    ComponentConflict,
)
from pint.models.timing_model import PhaseComponent, Component
from pint.models.parameter import floatParameter
from pint.utils import split_prefixed_name, PrefixError


class SimpleModel(PhaseComponent):
    """Very simple test model component
    """

    register = True
    category = "simple_test"

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.add_param(floatParameter(name="TESTPARAM", value=0.0, unit="s"))


class SubsetModel(PhaseComponent):
    """Test model component hosting the parameters which are a subset of spindown.
    """

    register = False  # This has to be false, otherwrise all test will fail.
    category = "simple_test"

    def __init__(self):
        super(SubsetModel, self).__init__()
        self.add_param(floatParameter(name="F0", value=0.0, unit="1/s"))
        self.add_param(floatParameter(name="F1", value=0.0, unit="1/s^2"))


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


def test_model_builder_class():
    """Test if model builder collected components information correctly
    """
    mb = ModelBuilder()
    category = mb.category_component_map
    assert len(mb.param_component_map["PX"]) == len(category["astrometry"])
    assert len(mb.component_category_map) == len(mb.components)
    assert len(mb.param_alias_map) == len(mb.param_component_map)
    # test for new components
    assert "SimpleModel" in mb.components
    simple_comp = mb.components["SimpleModel"]
    simple_comp.add_param(
        floatParameter(name="TESTPARAM2", aliases=["F0"], value=0.0, unit="s")
    )


def test_aliases_mapping():
    """Test if aliases gets mapped correclty
    """
    mb = ModelBuilder()
    # all alases should be mapped to the components
    assert len(mb.param_alias_map) == len(mb.param_component_map)

    # Test if the param_alias_map is passed by pointer
    # Testing the private function for building the aliases map
    mb._add_alias_to_map("TESTAX", "TESTAXX", mb.param_alias_map)
    assert "TESTAX" in mb.param_alias_map
    # Test existing entry
    # When adding an existing alias to the map. The mapped value should be the
    # same, otherwrise it will fail.
    mb._add_alias_to_map("F0", "F0", mb.param_alias_map)
    assert mb.param_alias_map["F0"] == "F0"
    # Test repeatable_params with differnt indices.
    for rp in mb.repeatable_param:
        pint_par = mb.alias_to_pint_param(rp)
        cp = mb.param_component_map[pint_par][0]
        pint_par_obj = getattr(mb.components[cp], pint_par)
        try:
            prefix, id, ids = split_prefixed_name(rp)
        except PrefixError:
            prefix = rp

        new_idx_par = prefix + "2"
        assert mb.alias_to_pint_param(new_idx_par) == pint_par_obj.prefix + "2"
        new_idx_par = prefix + "55"
        assert mb.alias_to_pint_param(new_idx_par) == pint_par_obj.prefix + "55"
        # Test all aliases
        for als in pint_par_obj.aliases:
            assert mb.alias_to_pint_param(als) == pint_par_obj.name
            try:
                als_prefix, id, ids = split_prefixed_name(als)
            except PrefixError:
                als_prefix = als
        assert mb.alias_to_pint_param(als_prefix + "2") == pint_par_obj.prefix + "2"
        assert mb.alias_to_pint_param(als_prefix + "55") == pint_par_obj.prefix + "55"


def test_conflict_alias():
    """Test if model builder detects the alais conflict.
    """
    mb = ModelBuilder()
    # Test conflict parameter alias name
    with pytest.raises(ConflictAliasError):
        _ = mb._add_alias_to_map("F0", "F1", mb.param_alias_map)
    # Define conflict alais from component class
    class SimpleModel2(PhaseComponent):
        """Very simple test model component
        """

        register = True
        category = "simple_test"

        def __init__(self):
            super(SimpleModel2, self).__init__()
            self.add_param(
                floatParameter(name="TESTPARAMF0", aliases=["F0"], value=0.0, unit="s")
            )

    with pytest.raises(ConflictAliasError):
        mb2 = ModelBuilder()
    del Component.component_types["SimpleModel2"]


def test_overlap_component(simple_model_overlap, simple_model_alias_overlap):
    """Test if model builder detects the overlap component correctly.
    """
    mb = ModelBuilder()
    # Test overlap
    overlap = mb._get_component_param_overlap(simple_model_overlap)
    assert "Spindown" in overlap.keys()
    assert overlap["Spindown"][0] == set(["F0"])
    # Only one over lap parameter F0
    # Since the _get_component_param_overlap returns non-overlap part,
    # we test if the non-overlap number makes sense.
    assert overlap["Spindown"][1] == len(simple_model_overlap.params) - 1
    assert overlap["Spindown"][2] == len(mb.components["Spindown"].params) - 1

    a_overlap = mb._get_component_param_overlap(simple_model_alias_overlap)
    assert a_overlap["Spindown"][0] == set(["F0"])
    assert a_overlap["Spindown"][1] == len(simple_model_alias_overlap.params) - 1
    assert a_overlap["Spindown"][2] == len(mb.components["Spindown"].params) - 1
    assert a_overlap["AstrometryEcliptic"][0] == set(["ELONG"])
    assert (
        a_overlap["AstrometryEcliptic"][1] == len(simple_model_alias_overlap.params) - 1
    )
    assert (
        a_overlap["AstrometryEcliptic"][2]
        == len(mb.components["AstrometryEcliptic"].params) - 1
    )


def test_subset_component(sub_set_model):
    """Test if model builder detects the subset component.
    """
    mb = ModelBuilder()
    # Test subset
    superset = mb._is_subset_component(sub_set_model)
    assert superset == "Spindown"


def test_model_from_par():
    """Test Get model from test par file.
    """
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
    param_inpar, repeat = mb.parse_parfile(io.StringIO(test_par1))
    assert len(param_inpar) == 60
    assert len(repeat) == 4
    comps, conflict, unknown_param = mb.choose_model(param_inpar)
    tm = mb(io.StringIO(test_par1))
    assert len(comps) == 14


def test_model_from_par_hassubset():
    """Test Get model from test par file with a subset component.
    """
    # Define a subset parameter model that is registered. So the metaclass can
    # catch it.
    class SubsetModel2(PhaseComponent):
        """Test model component hosting the parameters which are a subset of spindown.
        """

        register = True
        category = "simple_test"

        def __init__(self):
            super(SubsetModel2, self).__init__()
            self.add_param(floatParameter(name="F0", value=0.0, unit="1/s"))
            self.add_param(floatParameter(name="F1", value=0.0, unit="1/s^2"))

    with pytest.raises(ComponentConflict):
        mb = ModelBuilder()
    # Have to remove the SubsetModel2, since it will fail other tests.
    del Component.component_types["SubsetModel2"]
