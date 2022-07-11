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
    DM1            3.0000787439202574753e-05 1  0.00001693023051666952   
    DM2            7.1817346006345240866e-05 1  0.00006204477758539090   
    PMRA           -12.470619860337469966    1  0.50554723104847243409   
    PMDEC          -6.2676364069765152904    1  0.98468947896300551559   
    PX             0.47462929231991178613    1  0.33509329620827088547   
    BINARY         ELL1
    PB             0.13879914244858396754    1  0.00000000003514075083   
    A1             0.034841158415224894973   1  0.00000012173038389247   
    TASC           56178.804891768506529     1  0.00000007765191894742   
    EPS1           1.6508830631753595232e-05 1  0.00000477568412215803   
    EPS2           3.9656838708709247373e-06 1  0.00000458951091435993   
    TZRMJD         56166.271106550830051       
    TZRFRQ         1499.5310059999999339    
    TZRSITE        ao   
    FD1            4.250598511058125771e-05  1  0.00000052131001364731   
    TRES           2.560        
    NE_SW          4                           
    CLK            TT(BIPM2015)
    MODE 1
    UNITS          TDB
    TIMEEPH        FB90
    DILATEFREQ     N
    PLANET_SHAPIRO N
    CORRECT_TROPOSPHERE  N
    EPHEM          DE436
    NITS           1
    NTOA           4373
    CHI2R          1.0173 4355
    JUMP -fe 430 0 0
    JUMP -fe L-wide 1.8531408320718e-05 1
    TNEF -group 430_ASP 1.0389
    TNEF -group 430_PUPPI 0.985946
    TNEF -group L-wide_ASP 1.13428
    TNEF -group L-wide_PUPPI 1.30189
    TNEQ -group 430_ASP -8.77109
    TNEQ -group 430_PUPPI -8.26324
    TNEQ -group L-wide_ASP -8.55545
    TNEQ -group L-wide_PUPPI -7.86103
    TNECORR -group 430_PUPPI 0.00490558
    TNRedAmp -13.3087
    TNRedGam 0.125393
    TNRedC 14
"""

@pytest.fixture()
def modelJ0023p0923():
    return mb.get_model(io.StringIO(parfile_contents))

def test_read_PLRedNoise(modelJ0023p0923):
    assert ('PLRedNoise' in modelJ0023p0923.components
                and modelJ0023p0923.components['PLRedNoise'] in modelJ0023p0923.NoiseComponent_list
                and hasattr(modelJ0023p0923, 'TNREDAMP')
                and hasattr(modelJ0023p0923, 'TNREDGAM')
                and hasattr(modelJ0023p0923, 'TNREDC')
            ), "PLRedNoise test failed."

if __name__ == "__main__":
    pass

