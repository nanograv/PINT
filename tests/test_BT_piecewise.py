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
import unittest
from io import StringIO
from pylab import *
import pytest
import pint.residuals
import pint.simulation


@pytest.fixture
def build_model():
    #builds a J1023+0038-like model with no pieces
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


@pytest.fixture()
def make_toas_to_go_with_two_piece_model(build_piecewise_model_with_two_pieces):
    #makes toas to go with the two non-overlapping, complete coverage model
    m_piecewise = deepcopy(build_piecewise_model_with_two_pieces)
    m_piecewise.setup()
    toas = pint.simulation.make_fake_toas_uniform(m_piecewise.START.value+1,m_piecewise.FINISH.value-1,20,m_piecewise) #slightly within group edges to make toas unambiguously contained within groups
    return toas


def get_toa_group_indexes(model,toas):
    #returns array of group indexes associated with each toa. (i.e. which group is each toa in)
    model.setup()
    index = model.which_group_is_toa_in(toas)
    return index


def get_number_of_groups(model):
    #returns number of groups
    model.setup()
    number_of_groups = model.get_number_of_groups()
    print(model.as_parfile())
    print(model.get_number_of_groups())
    return number_of_groups


@pytest.fixture()
def build_piecewise_model_with_two_pieces(build_model):
    #takes the basic model frame and adds 2 non-ovelerlapping pieces to it
    piecewise_model = deepcopy(build_model)
    piecewise_model.remove_range(0)
    piecewise_model.setup()
    spacing = (piecewise_model.FINISH.value-piecewise_model.START.value)/2
    for i in range(0,2):
        piecewise_model.add_group_range(piecewise_model.START.value+spacing*i,piecewise_model.START.value+spacing*(i+1),j=i)
        piecewise_model.add_piecewise_param("A1","ls",piecewise_model.A1.value+i*1e-5,i)
        piecewise_model.add_piecewise_param("T0","d",piecewise_model.T0.value+i*1e-5,i)
    return piecewise_model


def test_add_piecewise_parameter(build_model):
    #test: see if the model can be reproduced after piecewise parameters have been added,
    #checks by comparing parameter keys in both the old and new file. Should have the number of matches = number of parameters
    m_piecewise=deepcopy(build_model)
    n=10
    m_piecewise.remove_range(0)
    for i in range(0,n):
        m_piecewise.add_group_range(m_piecewise.START.value+501*i,m_piecewise.START.value+1000*i,j=i)
        m_piecewise.add_piecewise_param("A1","ls",m_piecewise.A1.value+1,i)
        m_piecewise.add_piecewise_param("T0","d",m_piecewise.T0.value,i)
    m3=get_model(StringIO(m_piecewise.as_parfile()))
    param_dict = m_piecewise.get_params_dict(which='all')
    copy_param_dict = m3.get_params_dict(which='all')
    number_of_keys = 0
    comparison = 0
    for key, value in param_dict.items():
        number_of_keys = number_of_keys + 1 #iterates up to total number of keys
        for copy_key, copy_value in param_dict.items():
            if key == copy_key:        #search both pars for identical keys 
                if value.quantity == copy_value.quantity:
                    comparison = comparison + 1 #iterates up to all matched keys, which should be all, hence total number of keys
    assert comparison == number_of_keys #assert theyre the same length
        

def test_get_number_of_groups(build_piecewise_model_with_two_pieces):
    #test to make sure number of groups matches with number of added piecewise parameters
    m_piecewise = build_piecewise_model_with_two_pieces
    number_of_groups = get_number_of_groups(m_piecewise)
    assert number_of_groups == 2
        
def test_group_assignment_toas_unambiguously_within_group(build_piecewise_model_with_two_pieces, make_toas_to_go_with_two_piece_model):
    #test to see if the group, for one toa per group for 5 groups, that the BT_piecewise.print_toas_in_group functions as intended.
    #operates by sorting the toas by MJD compared against a groups upper/lower edge.
    #operates with np.searchsorted so for 1 toa per group, each toa should be uniquely indexed after/before the lower/upper edge
    index = get_toa_group_indexes(build_piecewise_model_with_two_pieces , make_toas_to_go_with_two_piece_model)
    print(index)
    should_be_ten_toas_in_each_group = [np.unique(index,return_counts = True)[1][0],np.unique(index,return_counts = True)[1][1]]
    expected_toas_in_each_group = [10,10]
    is_there_ten_toas_per_group = np.array_equal(should_be_ten_toas_in_each_group , expected_toas_in_each_group)
    assert is_there_ten_toas_per_group
    

@pytest.mark.parametrize("param",["A1X","T0X"])    
def test_paramX_per_toa_matches_corresponding_model_value(param, build_piecewise_model_with_two_pieces, make_toas_to_go_with_two_piece_model): 
    #Testing the correct piecewise parameters are being assigned to each toa.
    #Operates on the piecewise_parameter_from_information_array function. Requires group_index fn to be run so we have an array of length(ntoas), filled with information on which group a toa belongs to. 
    #Uses this array to apply T0X_i/A1X_i to corresponding indexes from group_index fn call. i.e. for T0X_i,T0X_j,T0X_k values and group_index return: [i,j,k] the output would be [T0X_i,T0X_j,T0X_k] 
    m_piecewise = build_piecewise_model_with_two_pieces
    toa = make_toas_to_go_with_two_piece_model
    m_piecewise.setup()
    rs = pint.residuals.Residuals(toa,m_piecewise)
    if param == "A1X":
        
        paramX_per_toa = m_piecewise.get_A1Xs_associated_with_toas(toa)
        test_val = [m_piecewise.A1X_0000.quantity,m_piecewise.A1X_0001.quantity]
        should_toa_reference_piecewise_parameter = [m_piecewise.does_toa_reference_piecewise_parameter(toa,"A1X_0000"),m_piecewise.does_toa_reference_piecewise_parameter(toa,"A1X_0001")]
    elif param == "T0X":
        paramX_per_toa = m_piecewise.get_T0Xs_associated_with_toas(toa)
        test_val = [m_piecewise.T0X_0000.quantity,m_piecewise.T0X_0001.quantity]
        should_toa_reference_piecewise_parameter = [m_piecewise.does_toa_reference_piecewise_parameter(toa,"T0X_0000"),m_piecewise.does_toa_reference_piecewise_parameter(toa,"T0X_0001")]
    
    do_toas_reference_first_piecewise_parameter = np.isclose((paramX_per_toa.value-test_val[0].value),0,atol=1e-6,rtol=0)
    do_toas_reference_second_piecewise_parameter = np.isclose((paramX_per_toa.value-test_val[1].value),0,atol=1e-6,rtol=0)
    do_toas_reference_piecewise_parameter = [do_toas_reference_first_piecewise_parameter,do_toas_reference_second_piecewise_parameter]
    do_arrays_match = np.array_equal(do_toas_reference_piecewise_parameter,should_toa_reference_piecewise_parameter)
    assert do_arrays_match
    

        
def test_problematic_group_indexes_and_ranges(build_model):
    #Test to flag issues with problematic group indexes
    m_piecewise = build_model
    with pytest.raises(Exception):
        assert m_piecewise.add_group_range(m_piecewise.START.value,m_piecewise.FINISH.value, j=-1)
        assert m_piecewise.add_group_range(m_piecewise.FINISH.value,m_piecewise.START.value, j=1)
        assert m_piecewise.add_group_range(m_piecewise.START.value,m_piecewise.FINISH.value, j=10000)
    

def add_groups_and_make_toas(build_model,build_piecewise_model_with_two_pieces,param):
    #function to build the models for specific edge cases
    spacing = (build_model.FINISH.value-build_model.START.value)/2
    if param == "non-overlapping complete group coverage":
        model = build_piecewise_model_with_two_pieces
        model.setup()
        #print(model.as_parfile())
        toas = make_generic_toas(model) 
        return model,toas
    elif param == "overlapping groups":
        model2 = build_model
        model2.remove_range(0)
        model2.add_group_range(model2.START.value-1, model2.START.value + spacing + 100, j = 0)
        model2.add_group_range(model2.START.value+spacing-100, model2.FINISH.value + 1, j = 1)
        model2.add_piecewise_param("T0","d",model2.T0.value+1e-5,0)
        model2.add_piecewise_param("T0","d",model2.T0.value+2e-5,1)
        model2.add_piecewise_param("A1","ls",model2.A1.value+1e-5,0)
        model2.add_piecewise_param("A1","ls",model2.A1.value+2e-5,1)
        model2.setup()
        toas = make_generic_toas(model2)
        return model2,toas
    elif param == "non-complete group coverage":
        model3 = build_model
        model3.remove_range(0)
        model3.add_group_range(model3.START.value + spacing , model3.FINISH.value + 1 , j = 0)
        model3.add_piecewise_param("T0","d",model3.T0.value+1e-5,0)
        model3.setup()
        toas = make_generic_toas(model3)
        return model3,toas
    
def make_generic_toas(model):
    #makes toas to go with the edge cases
    return pint.simulation.make_fake_toas_uniform(model.START.value , model.FINISH.value-1 , 20 , model)

def convert_int_into_index(i):
    #converts i to 4-digit integer: 1->0001, 1010->1010
    return f"{int(i):04d}" 
    

def check_if_in_piece(model,toas):
    #Tests if the in_piece fn works. in_piece is used in the fitter to only apply the adjustment to toas inside a group. Returns truth array, true if toa lies in the T0X_i we are interested in and false elsewhere
    #Check to see if these are being allocated correctly.
    is_toa_in_each_group = []
    for i in range(model.get_number_of_groups()):
        par = f"T0X_{convert_int_into_index(i)}"
        is_toa_in_each_group.append(model.does_toa_reference_piecewise_parameter(toas,par))
    return is_toa_in_each_group


def return_truth_array_based_on_group_boundaries(model,barycentric_toa):
    boundaries = model.get_group_boundaries()
    upper_edge_of_lower_group = boundaries[1][0]
    lower_edge_of_upper_group = boundaries[0][1]
    truth_array_comparison = [[barycentric_toa.value<=upper_edge_of_lower_group],[barycentric_toa.value>=lower_edge_of_upper_group]]
    return truth_array_comparison

@pytest.mark.parametrize("param",["non-overlapping complete group coverage","overlapping groups", "non-complete group coverage"])
def test_does_toa_lie_in_group(build_model,build_piecewise_model_with_two_pieces,param):
    m_piecewise,toas = add_groups_and_make_toas(build_model,build_piecewise_model_with_two_pieces,param)
    is_toa_in_each_group = check_if_in_piece(m_piecewise,toas)
    barycentric_toa = m_piecewise._parent.get_barycentric_toas(toas)
    if param == "non-overlapping complete group coverage":
        #Logic is toas lie in group 0:10 should all be True(/=1) [in the piece]. And false for toas out of the group. (T0X_0000 being used) 
        #and vice versa for the T0X_0001
        
        are_toas_within_group_boundaries_mjd_method_per_parameter = return_truth_array_based_on_group_boundaries(m_piecewise,barycentric_toa)
        
        is_toa_in_each_group = np.concatenate((is_toa_in_each_group[0],is_toa_in_each_group[1]))
        
        are_toas_within_group_boundaries_mjd_method_per_parameter = np.concatenate((are_toas_within_group_boundaries_mjd_method_per_parameter[0][0] , are_toas_within_group_boundaries_mjd_method_per_parameter[1][0]))
        
        does_in_piece_and_mjd_method_agree = np.array_equal(are_toas_within_group_boundaries_mjd_method_per_parameter,is_toa_in_each_group)
        
        assert does_in_piece_and_mjd_method_agree
    
    elif param == "overlapping groups":
        #Test to check if the central 2 toas remain unallocated to a group after checking both groups in_piece property.
        #i.e. [T,T,F,F,F,F]+[F,F,F,F,T,T] = [T,T,F,F,T,T]. 
        #Reliant on the above test passing as doesn't catch [T,T,F,F,T,T]+[T,T,F,F,T,T] (this output would mean in_piece would just be checking if a toa belonged to any group)
        
        are_toas_within_group_boundaries_mjd_method_per_parameter = return_truth_array_based_on_group_boundaries(m_piecewise,barycentric_toa)
        
        where_toas_should_be_in_group_1 = np.where(are_toas_within_group_boundaries_mjd_method_per_parameter[0][0]!=are_toas_within_group_boundaries_mjd_method_per_parameter[1][0],are_toas_within_group_boundaries_mjd_method_per_parameter[0][0],are_toas_within_group_boundaries_mjd_method_per_parameter[0][0]!=are_toas_within_group_boundaries_mjd_method_per_parameter[1][0])        
        
        where_toas_should_be_in_group_2 = np.where(are_toas_within_group_boundaries_mjd_method_per_parameter[0][0]!=are_toas_within_group_boundaries_mjd_method_per_parameter[1][0] , are_toas_within_group_boundaries_mjd_method_per_parameter[1][0] , are_toas_within_group_boundaries_mjd_method_per_parameter[0][0]!=are_toas_within_group_boundaries_mjd_method_per_parameter[1][0])
        
        is_toa_in_each_group = [is_toa_in_each_group[0],is_toa_in_each_group[1]]
        
        are_toas_within_group_boundaries_mjd_method_per_parameter = [where_toas_should_be_in_group_1,where_toas_should_be_in_group_2]
        
        does_in_piece_and_mjd_method_agree = np.array_equal(are_toas_within_group_boundaries_mjd_method_per_parameter, is_toa_in_each_group)
        
        assert does_in_piece_and_mjd_method_agree
    
    elif param == "non-complete group coverage":
        #This test is performed to make sure toas that shouldn't be in any group are correctly being flagged. Only later half of toas should be in group. Distinct from the overlapping group test since its not ambiguous which group they belong to since they aren't in a group
        
        boundaries = m_piecewise.get_group_boundaries()
        
        lower_edge_of_group = boundaries[0][0]
        are_toas_above_group_edge = [barycentric_toa.value>=lower_edge_of_group]
        
        do_in_piece_and_mjd_methods_of_assigning_groups_agree = np.array_equal(is_toa_in_each_group,are_toas_above_group_edge)
        
        assert do_in_piece_and_mjd_methods_of_assigning_groups_agree
        
    

def add_offset_in_model_parameter(indexes,param,model):
    m_piecewise_temp = deepcopy(model)
    parameter_string = f"{param}_{convert_int_into_index(indexes)}"
    if hasattr(m_piecewise_temp,parameter_string):
        delta = getattr(m_piecewise_temp,parameter_string).value+1e-5
        getattr(m_piecewise_temp , parameter_string).value = delta
        m_piecewise_temp.setup()
    else:
        parameter_string = param[0:2]
        getattr(m_piecewise_temp,parameter_string).value = getattr(m_piecewise_temp,parameter_string).value+1e-5
        m_piecewise_temp.setup()
    return m_piecewise_temp
    
    
@pytest.mark.parametrize("param, index",[("T0X",0),("T0X",1)])
def test_residuals_in_groups_respond_to_changes_in_corresponding_piecewise_parameter(build_model,build_piecewise_model_with_two_pieces,param,index):
    m_piecewise,toas = add_groups_and_make_toas(build_model,build_piecewise_model_with_two_pieces,"overlapping groups")
    #m_piecewise.setup()
    rs_value = pint.residuals.Residuals(toas,m_piecewise,subtract_mean=False).resids_value
    parameter_string = f"{param}_{convert_int_into_index(index)}"

    output_param = []
    m_piecewise_temp = add_offset_in_model_parameter(index,param,m_piecewise)
 
    rs_temp = pint.residuals.Residuals(toas,m_piecewise_temp,subtract_mean=False).resids_value
    have_residuals_changed = np.invert(rs_temp == rs_value)
    should_residuals_change = m_piecewise.does_toa_reference_piecewise_parameter(toas,parameter_string)
    are_correct_residuals_responding = np.array_equal(have_residuals_changed,should_residuals_change)
    

    assert are_correct_residuals_responding
    
    
@pytest.mark.parametrize("param, index",[("T0X",0),("T0X",1),("T0X",-1),("A1X",0),("A1X",1),("A1X",-1)])
#assert array equal truth_array not (~) in_piece
def test_d_delay_in_groups_respond_to_changes_in_corresponding_piecewise_parameter(build_model,build_piecewise_model_with_two_pieces,param,index):
    m_piecewise,toas = add_groups_and_make_toas(build_model,build_piecewise_model_with_two_pieces,"overlapping groups")
    m_piecewise_temp = add_offset_in_model_parameter(index,param,m_piecewise)
    parameter_string = f"{param}_{convert_int_into_index(index)}"
    if index==-1:
        is_d_delay_changing = np.invert(np.isclose(m_piecewise_temp.d_binary_delay_d_xxxx(toas,param[0:2],None).value,0,atol=1e-11,rtol=0))
    else:
        is_d_delay_changing = np.invert(np.isclose(m_piecewise_temp.d_binary_delay_d_xxxx(toas,parameter_string,None).value,0,atol=1e-11,rtol=0))
    should_d_delay_be_changing = m_piecewise.does_toa_reference_piecewise_parameter(toas,parameter_string)
    #assert toas that are in the group/gap have some non-zero delay derivative
    both_arrays_should_be_equal = np.array_equal(is_d_delay_changing,should_d_delay_be_changing)
    assert both_arrays_should_be_equal
    

def add_relative_offset_for_derivatives(index,param,model,offset_size,plus=True):
    m_piecewise_temp = deepcopy(model)
    parameter_string = f"{param}_{convert_int_into_index(index)}"
    offset_size = offset_size.value
    if plus is True:
        if hasattr(m_piecewise_temp,parameter_string):
            delta = getattr(m_piecewise_temp,parameter_string).value+offset_size
            getattr(m_piecewise_temp , parameter_string).value = delta
            m_piecewise_temp.setup()
        else:
            parameter_string = param[0:2]
            getattr(m_piecewise_temp,parameter_string).value = getattr(m_piecewise_temp,parameter_string).value+offset_size
            m_piecewise_temp.setup()
        return m_piecewise_temp
    else:
        if hasattr(m_piecewise_temp,parameter_string):
            delta = getattr(m_piecewise_temp,parameter_string).value-offset_size
            getattr(m_piecewise_temp , parameter_string).value = delta
            m_piecewise_temp.setup()
        else:
            parameter_string = param[0:2]
            getattr(m_piecewise_temp,parameter_string).value = getattr(m_piecewise_temp,parameter_string).value-offset_size
            m_piecewise_temp.setup()
        return m_piecewise_temp
    

def get_d_delay_d_xxxx(toas,model,parameter_string):
        rs_temp = pint.residuals.Residuals(toas,model,subtract_mean=False)
        d_delay_temp = rs_temp.model.d_binary_delay_d_xxxx(toas,parameter_string,acc_delay = None)
        return d_delay_temp    
    
    
#@pytest.mark.parametrize("param, index, offset_size",[("T0X",0, 1e-5*u.d),#("T0X",-1, 1e-5*u.d),("A1X",0, 1e-5*ls),("A1X",-1, 1e-5*ls)])    
#def test_d_delay_is_producing_correct_numbers(build_model,build_piecewise_model_with_two_pieces,param,index,offset_size):
#    m_piecewise,toas = add_groups_and_make_toas(build_model,build_piecewise_model_with_two_pieces,"overlapping groups")
#    m_piecewise_plus_offset = add_relative_offset_for_derivatives(index,param,m_piecewise,offset_size,plus = True)
#    m_piecewise_minus_offset = add_relative_offset_for_derivatives(index,param,m_piecewise,offset_size,plus = False)
    
#    parameter_string = f"{param}_{convert_int_into_index(index)}"
#    if parameter_string[-4] == "-":
#        parameter_string = parameter_string[0:2]
    
#    derivative_unit = offset_size.unit
    
#    m_piecewise.setup()
#    m_piecewise_plus_offset.setup()
#    m_piecewise_minus_offset.setup()
    
    #plus_d_delay = get_d_delay_d_xxxx(toas,m_piecewise_plus_offset,parameter_string)
    #minus_d_delay = get_d_delay_d_xxxx(toas,m_piecewise_minus_offset,parameter_string)
    #no_offset_d_delay = get_d_delay_d_xxxx(toas,m_piecewise,parameter_string)
    
    #average_gradient_from_plus_minus_offset = ((plus_d_delay+minus_d_delay).to(u.s/derivative_unit, equivalencies = u.dimensionless_angles())/(2*offset_size))
    
    #d_delay_is_non_zero = np.invert(np.isclose(average_gradient_from_plus_minus_offset.value,0,atol = 1e-5,rtol=0))
    #where_d_delays_should_be_non_zero = m_piecewise.does_toa_reference_piecewise_parameter(toas,parameter_string)
    
    #no_offset_gradient = (no_offset_d_delay.to(u.s/derivative_unit, equivalencies = u.dimensionless_angles())/offset_size)
    #are_they_close = np.isclose(average_gradient_from_plus_minus_offset,no_offset_gradient,atol = 1,rtol = 0)
    #every_d_delay_should_be_close = np.ones_like(are_they_close)
    
    #conditions_for_pass = [where_d_delays_should_be_non_zero, every_d_delay_should_be_close]
    #test_against_conditions = [d_delay_is_non_zero , are_they_close]
    
    
    #print(f"Array 1: {where_d_delays_should_be_non_zero}")
    #print(f"Array 2: {d_delay_is_non_zero}")
    
    
    #assert np.array_equal(test_against_conditions,conditions_for_pass)