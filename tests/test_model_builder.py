"""Test model builder using variance of parfile input"""

from glob import glob
from os.path import basename, join
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
from pint.models.timing_model import PhaseComponent
from pint.models.parameter import floatParameter
from pint.utils import split_prefixed_name, PrefixError


test_par1 ="""
PSR              B1855+09
#RAJ             18:57:36.3932884         1  0.00002602730280675029
#DECJ           +09:43:17.29196           1  0.00078789485676919773
#PMRA           -2.5054345161030380639    1  0.03104958261053317181
#PMDEC          -5.4974558631993817232    1  0.06348008663748286318
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


class SimpleModel(PhaseComponent):
    register = True
    categore = "simple_test"
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.add_param(floatParameter(name='TESTPARAM', value=0.0, unit='s'))

# @pytest.fixture
# def tmp_dir(tmpdir):
#     yield str(tmpdir)
def test_model_builder_class():
    mb = ModelBuilder()
    categore = mb.category_component_map
    assert len(mb.param_component_map['PX']) == len(categore['astrometry'])
    assert len(mb.component_category_map) == len(mb.components)
    assert len(mb.param_alias_map) == len(mb.param_component_map)
    # test for new components
    assert 'SimpleModel' in mb.components
    simple_comp = mb.components['SimpleModel']
    simple_comp.add_param(floatParameter(name='TESTPARAM2', aliases=['F0'],
        value=0.0, unit='s'))
    mb.param_alias_map

def test_aliases_mapping():
    mb = ModelBuilder()
    assert len(mb.param_alias_map) == len(mb.param_component_map)

    # Test if the param_alias_map is passed by pointer
    _ = mb._add_alias_to_map("TESTAX", "TESTAXX", mb.param_alias_map)
    assert "TESTAX" in mb.param_alias_map
    # Test existing entry
    _ = mb._add_alias_to_map("F0", "F0", mb.param_alias_map)
    assert mb.param_alias_map['F0'] == 'F0'
    with pytest.raises(ConflictAliasError):
        _ = mb._add_alias_to_map("F0", "F1", mb.param_alias_map)
    for rp in mb.repeatable_param:
        pint_par = mb.alias_to_pint_param(rp)
        cp = mb.param_component_map[pint_par][0]
        pint_par_obj = getattr(mb.components[cp], pint_par)
        try:
            prefix, id, ids = split_prefixed_name(rp)
        except PrefixError:
            prefix = rp

        new_idx_par = prefix + '2'
        assert mb.alias_to_pint_param(new_idx_par) == pint_par_obj.prefix + '2'
        new_idx_par = prefix + '55'
        assert mb.alias_to_pint_param(new_idx_par) == pint_par_obj.prefix + '55'
        # Test aliases
        for als in pint_par_obj.aliases:
            assert mb.alias_to_pint_param(als) == pint_par_obj.name
            try:
                als_prefix, id, ids = split_prefixed_name(als)
            except PrefixError:
                als_prefix = als
        assert mb.alias_to_pint_param(als_prefix + '2') == pint_par_obj.prefix + '2'
        assert mb.alias_to_pint_param(als_prefix + '55') == pint_par_obj.prefix + '55'
# def test_new_component(component):
#     mb = ModelBuilder()
#     # Test overlap
#     overlap = mb._get_component_param_overlap(component)
#     # Test subset
#     subset = mb._is_subset_component(component)


def test_model_par():
    mb = ModelBuilder()
    param_inpar, repeat = mb.parse_parfile(io.StringIO(test_par1))
    assert len(param_inpar) == 60
    assert len(repeat) == 4
    comps, conflict, unknown_param = mb.choose_model(param_inpar)
    tm = mb(io.StringIO(test_par1))
    assert len(comps) == 14


    #tm = mb(test_par1)
