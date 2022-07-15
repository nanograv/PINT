"""Tests to ensure that ModelBuilder is able to read PLRedNoise 
component properly."""

import io
import pytest

import pint.models.model_builder as mb

parfile_contents = """
    PSRJ           J0023+0923
    RAJ             00:23:16.8790858         1  0.00002408141295805134   
    DECJ           +09:23:23.86936           1  0.00082010713730773120   
    F0             327.84702062954047136     1  0.00000000000295205483   
    F1             -1.2278326306812866375e-15 1  3.8219244605614075223e-19
    PEPOCH         56199.999797564144902       
    POSEPOCH       56199.999797564144902       
    DMEPOCH        56200                       
    DM             14.327978186774068347     1  0.00006751663559857748   
    BINARY         ELL1
    PB             0.13879914244858396754    1  0.00000000003514075083   
    A1             0.034841158415224894973   1  0.00000012173038389247   
    TASC           56178.804891768506529     1  0.00000007765191894742   
    EPS1           1.6508830631753595232e-05 1  0.00000477568412215803   
    EPS2           3.9656838708709247373e-06 1  0.00000458951091435993   
    CLK            TT(BIPM2015)
    MODE 1
    UNITS          TDB
    TIMEEPH        FB90
    DILATEFREQ     N
    PLANET_SHAPIRO N
    CORRECT_TROPOSPHERE  N
    EPHEM          DE436
    JUMP -fe 430 0 0
    TNEF -group 430_ASP 1.0389
    TNEQ -group 430_ASP -8.77109
    TNECORR -group 430_PUPPI 0.00490558
    TNRedAmp -13.3087
    TNRedGam 0.125393
    TNRedC 14
"""


@pytest.fixture()
def modelJ0023p0923():
    return mb.get_model(io.StringIO(parfile_contents))


def test_read_PLRedNoise_component(modelJ0023p0923):
    assert "PLRedNoise" in modelJ0023p0923.components


def test_read_PLRedNoise_component_type(modelJ0023p0923):
    assert (
        modelJ0023p0923.components["PLRedNoise"] in modelJ0023p0923.NoiseComponent_list
    )


def test_read_PLRedNoise_params(modelJ0023p0923):
    params = ["TNREDAMP", "TNREDGAM", "TNREDC"]
    for param in params:
        assert (
            hasattr(modelJ0023p0923, param)
            and getattr(modelJ0023p0923, param).quantity is not None
        )
