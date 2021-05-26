from pint.models import get_model
import pint.toa
import numpy as np
import pint.fitter
import astropy.units as u
from pint import ls
from copy import deepcopy
import pint.residuals
from astropy.time import Time
import pint.models.stand_alone_psr_binaries.BT_piecewise as BTpiecewise
import matplotlib.pyplot as plt
from io import StringIO
from pylab import *

def build_model():
    par_base = """
    PSR                             1023+0038
    TRACK                                  -3
    EPHEM                               DE421
    CLOCK                        TT(BIPM2019)
    START              54906.7327557001113774
    FINISH             56443.7950227385911112
    DILATEFREQ                              N
    DMDATA                                0.0
    NTOA                               4970.0
    CHI2                   14234223.733031187
    RAJ                     10:23:47.68719801
    DECJ                     0:38:40.84551000
    PMRA                                  0.0
    PMDEC                                 0.0
    PX                                    0.0
    POSEPOCH           54995.0000000000000000
    F0                   592.4214681620181649 0 
    F1              -2.4181868552365951721e-15 0
    PEPOCH             55352.6375960000000001
    PLANET_SHAPIRO                          N
    DM                  14.720549454038943568 1 
    DM1                                   0.0
    BINARY BT_piecewise
    PB                  0.1980963253021312476 1 
    PBDOT              -8.277368114929306e-11 1 
    A1                    0.34333468063634737 1 
    A1DOT               1.210496074329345e-13 1 
    ECC                                   0.0
    EDOT                                  0.0
    T0                 55361.5927961792717724 1 
    OM                                    0.0 0 
    OMDOT                                 0.0
    GAMMA                                 0.0
    TZRMJD             56330.1763862586450231
    TZRSITE                                 8
    TZRFRQ                           1547.743
    """
    model = get_model(StringIO(par_base))
    return model

#@pytest.mark.parameterize("n",[0,5,10])
def test_add_piecewise_parameter(n=10):
    m = build_model()
    m1=deepcopy(m)
    #Test to see if adding a piecewise parameter to a model can be read back in from a parfile
    for i in range(0,n):
        m1.add_group_range(m1.START.value+501*i,m1.START.value+1000*i,j=i)
        m1.add_piecewise_param("A1","ls",m1.A1.value+1,i)
        m1.add_piecewise_param("T0","d",m1.T0.value,i)
    m3=get_model(StringIO(m1.as_parfile()))
    param_dict = m1.get_params_dict(which='all')
    copy_param_dict = m3.get_params_dict(which='all')
    number_of_keys = 0
    comparison = 0
    for key, value in param_dict.items():
        number_of_keys = number_of_keys + 1 #oiterates up to total number of keys
        for copy_key, copy_value in param_dict.items():
            if key == copy_key:        #search both pars for identical keys 
                if value.quantity == copy_value.quantity:
                    comparison = comparison + 1 #iterates up to all matched keys, which should be all, hence total number of keys
    assert comparison == number_of_keys #assert theyre the same length
        


#@pytest.mark.parameterize("n",[0,10,100])
def test_toas_indexing_ordered_groups(n=10):
    m = build_model()
    m_piecewise = deepcopy(m)
    toa = pint.toa.make_fake_toas(m.START.value,m.FINISH.value,n,m)
    toa_spacing = (m_piecewise.FINISH.value-m_piecewise.START.value)/n
    for i in range(0,n):
        m_piecewise.add_group_range(m_piecewise.START.value+toa_spacing*i,m_piecewise.START.value+toa_spacing*(i+1),j=i)
        m_piecewise.add_piecewise_param("A1","ls",2*i,i)
        m_piecewise.add_piecewise_param("T0","d",50000+i,i)
    
    for i in range(0,n):
        string_base_num = f"{int(i):04d}" 
        string_base_T0 = f"T0X_{string_base_num}"
        string_base_A1 = f"A1X_{string_base_num}"
        assert hasattr(m_piecewise,string_base_T0) is True
        assert hasattr(m_piecewise,string_base_A1) is True
        assert abs(getattr(m_piecewise,string_base_T0).value-(50000+i))<=1e-6
        assert abs(getattr(m_piecewise,string_base_A1).value-2*i) <= 1e-6
    
