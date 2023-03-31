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
from pint.simulation import make_fake_toas_uniform


@pytest.fixture
def model_no_pieces(scope="session"):
    # builds a J1023+0038-like model with no pieces
    par_base = """
    PSR                             1023+0038
    TRACK                                  -3
    EPHEM                               DE421
    CLOCK                        TT(BIPM2019)
    START              55000.
    FINISH             55200.
    DILATEFREQ                              N
    RAJ                     10:23:47.68719801
    DECJ                     0:38:40.84551000
    POSEPOCH           54995.
    F0                   592.
    F1              -2.
    PEPOCH             55000.
    PLANET_SHAPIRO                          N
    DM                  14.
    BINARY BT_piecewise
    PB                  0.2
    PBDOT               0.0
    A1                    0.34333468063634737 1
    A1DOT               0.0
    ECC                                   0.0
    EDOT                                  0.0
    T0                 55000.
    TZRMJD             55000.
    TZRSITE                                 1
    """
    model = get_model(StringIO(par_base))
    # lurking bug: T0X_0000/A1X_0000 and boundaries are not automatically deleted on intialisation
    model.remove_range(0)
    return model


@pytest.fixture
def model_BT():
    # builds a J1023+0038-like model with no pieces
    par_base = """
    PSR                             1023+0038
    TRACK                                  -3
    EPHEM                               DE421
    CLOCK                        TT(BIPM2019)
    START              55000.
    FINISH             55200.
    DILATEFREQ                              N
    RAJ                     10:23:47.68719801
    DECJ                     0:38:40.84551000
    POSEPOCH           54995.
    F0                   592.
    F1              -2.
    PEPOCH             55000.
    PLANET_SHAPIRO                          N
    DM                  14.
    BINARY BT
    PB                  0.2
    PBDOT               0.0
    A1                    0.34333468063634737 1
    A1DOT               0.0
    ECC                                   0.0
    EDOT                                  0.0
    T0                 55000.
    TZRMJD             55000
    TZRSITE                                 1
    """
    model = get_model(StringIO(par_base))
    return model


@pytest.fixture()
def build_piecewise_model_with_one_T0_piece(model_no_pieces):
    # takes the basic model frame and adds 2 non-ovelerlapping pieces to it
    piecewise_model = deepcopy(model_no_pieces)
    lower_bound = [55000, 55100.00001]
    upper_bound = [55100, 55200]
    piecewise_model.add_group_range(lower_bound[0], upper_bound[0], j=0)
    piecewise_model.add_piecewise_param(
        "T0", 0, paramx=piecewise_model.T0.value + 1.0e-3
    )
    piecewise_model.lock_groups()
    return piecewise_model


@pytest.fixture()
def build_piecewise_model_with_one_A1_piece(model_no_pieces):
    # takes the basic model frame and adds 2 non-ovelerlapping pieces to it
    piecewise_model = deepcopy(model_no_pieces)
    lower_bound = [55000, 55100.00001]
    upper_bound = [55100, 55200]
    piecewise_model.add_group_range(lower_bound[0], upper_bound[0], j=0)
    piecewise_model.add_piecewise_param("A1", 0, paramx=piecewise_model.A1.value + 1e-3)
    piecewise_model.lock_groups()
    return piecewise_model


# fine function
@pytest.fixture()
def build_piecewise_model_with_two_pieces(model_no_pieces):
    # takes the basic model frame and adds 2 non-ovelerlapping pieces to it
    piecewise_model = model_no_pieces
    lower_bound = [55000, 55100.000000001]
    upper_bound = [55100, 55200]
    for i in range(len(lower_bound)):
        piecewise_model.add_group_range(lower_bound[i], upper_bound[i], j=i)
        piecewise_model.add_piecewise_param(
            "A1", i, paramx=piecewise_model.A1.value + (i + 1) * 1e-3
        )
        piecewise_model.add_piecewise_param(
            "T0", i, paramx=piecewise_model.T0.value + (i + 1) * 1e-3
        )
    piecewise_model.lock_groups()
    return piecewise_model


# fine function
@pytest.fixture()
def make_toas_to_go_with_two_piece_model(build_piecewise_model_with_two_pieces):
    # makes toas to go with the two non-overlapping, complete coverage model
    m_piecewise = build_piecewise_model_with_two_pieces
    lower_bound = [55000, 55100.00001]
    upper_bound = [55100, 55200]
    toas = make_fake_toas_uniform(
        lower_bound[0] + 1, upper_bound[1] - 1, 20, m_piecewise
    )  # slightly within group edges to make toas unambiguously contained within groups
    return toas


# fine function
def add_full_coverage_and_non_overlapping_groups_and_make_toas(
    model_no_pieces, build_piecewise_model_with_two_pieces
):
    # function to build the models for specific edge cases i.e. distinct groups where all toas fit exactly within a group
    model = build_piecewise_model_with_two_pieces
    toas = make_generic_toas(model, 55001, 55199)
    return model, toas


# fine function
def add_partial_coverage_groups_and_make_toas(build_piecewise_model_with_two_pieces):
    # function to build the models for specific edge cases i.e. if all toas don't fit exactly within any groups
    model3 = build_piecewise_model_with_two_pieces
    # known bug: if A1X exists it needs a partner T0X otherwise it breaks can freeze T0X for the time being, just needs a little thought
    model3.remove_range(0)
    model3.lock_groups()
    # make sure TOAs are within ranges
    toas = make_generic_toas(model3, 55001, 55199)
    return model3, toas


# fine function
def make_generic_toas(model, lower_bound, upper_bound):
    # makes toas to go with the edge cases
    return make_fake_toas_uniform(lower_bound, upper_bound, 20, model)


# fine function
def return_truth_array_based_on_group_boundaries(model, barycentric_toa):
    boundaries = model.get_group_boundaries()
    upper_edge_of_lower_group = boundaries[1][0]
    lower_edge_of_upper_group = boundaries[0][1]
    truth_array_comparison = [
        [barycentric_toa <= upper_edge_of_lower_group],
        [barycentric_toa >= lower_edge_of_upper_group],
    ]
    return truth_array_comparison


def add_offset_in_model_parameter(indexes, param, model):
    m_piecewise_temp = deepcopy(model)
    parameter_string = f"{param}_{int(indexes):04d}"
    if hasattr(m_piecewise_temp, parameter_string):
        delta = getattr(m_piecewise_temp, parameter_string).value + 1e-5
        getattr(m_piecewise_temp, parameter_string).value = delta
        m_piecewise_temp.setup()
    else:
        parameter_string = param[0:2]
        getattr(m_piecewise_temp, parameter_string).value = (
            getattr(m_piecewise_temp, parameter_string).value + 1e-5
        )
        m_piecewise_temp.setup()
    return m_piecewise_temp


def add_relative_offset_for_derivatives(index, param, model, offset_size, plus=True):
    m_piecewise_temp = deepcopy(model)
    parameter_string = f"{param}_{int(index):04d}"
    offset_size = offset_size.value
    if plus is True:
        if hasattr(m_piecewise_temp, parameter_string):
            delta = getattr(m_piecewise_temp, parameter_string).value + offset_size
            getattr(m_piecewise_temp, parameter_string).value = delta
            m_piecewise_temp.lock_groups()
    else:
        if hasattr(m_piecewise_temp, parameter_string):
            delta = getattr(m_piecewise_temp, parameter_string).value - offset_size
            getattr(m_piecewise_temp, parameter_string).value = delta
            m_piecewise_temp.lock_groups()
    return m_piecewise_temp


# fine function
def test_round_trips_to_parfile(model_no_pieces):
    # test: see if the model can be reproduced after piecewise parameters have been added,
    # checks by comparing parameter keys in both the old and new file. Should have the number of matches = number of parameters
    m_piecewise = model_no_pieces
    n = 10
    lower_bounds = [
        55050,
        55101,
        55151,
        55201,
        55251,
        55301,
        55351,
        55401,
        55451,
        55501,
    ]
    upper_bounds = [
        55100,
        55150,
        55200,
        55250,
        55300,
        55350,
        55400,
        55450,
        55500,
        55550,
    ]
    for i in range(0, n):
        m_piecewise.add_group_range(lower_bounds[i], upper_bounds[i], j=i)
        m_piecewise.add_piecewise_param("A1", i, paramx=m_piecewise.A1.value + i)
        m_piecewise.add_piecewise_param("T0", i, paramx=m_piecewise.T0.value + i)
    m_piecewise.lock_groups()
    m3 = get_model(StringIO(m_piecewise.as_parfile()))
    param_dict = m_piecewise.get_params_dict(which="all")
    copy_param_dict = m3.get_params_dict(which="all")
    number_of_keys = 0
    n_keys_identified = 0
    n_values_preserved = 0
    comparison = 0
    for key, value in param_dict.items():
        number_of_keys = number_of_keys + 1  # iterates up to total number of keys
        for copy_key, copy_value in copy_param_dict.items():
            if key == copy_key:  # search both pars for identical keys
                n_keys_identified = n_keys_identified + 1
                if type(value) == type(copy_value):
                    if value.value == copy_value.value:
                        n_values_preserved = n_values_preserved + 1
    assert n_keys_identified == number_of_keys
    assert n_values_preserved == number_of_keys


# fine function
def test_get_number_of_groups(build_piecewise_model_with_two_pieces):
    # test to make sure number of groups matches with number of added piecewise parameters
    m_piecewise = build_piecewise_model_with_two_pieces
    number_of_groups = m_piecewise.get_number_of_groups()
    assert number_of_groups == 2


# fine function
def test_group_assignment_toas_unambiguously_within_group(
    build_piecewise_model_with_two_pieces, make_toas_to_go_with_two_piece_model
):
    # test to see if the group, for one toa per group, that the BT_piecewise.print_toas_in_group functions as intended.
    # operates by sorting the toas by MJD compared against a groups upper/lower edge.
    # operates with np.searchsorted so for 1 toa per group, each toa should be uniquely indexed after/before the lower/upper edge
    model = build_piecewise_model_with_two_pieces
    index = model.which_group_is_toa_in(make_toas_to_go_with_two_piece_model)
    should_be_ten_toas_in_each_group = [
        np.unique(index, return_counts=True)[1][0],
        np.unique(index, return_counts=True)[1][1],
    ]
    expected_toas_in_each_group = [10, 10]
    is_there_ten_toas_per_group = np.testing.assert_array_equal(
        should_be_ten_toas_in_each_group, expected_toas_in_each_group
    )
    np.testing.assert_array_equal(
        should_be_ten_toas_in_each_group, expected_toas_in_each_group
    )


# fine function
@pytest.mark.parametrize("param", ["A1X", "T0X"])
def test_paramX_per_toa_matches_corresponding_model_value(
    param, build_piecewise_model_with_two_pieces, make_toas_to_go_with_two_piece_model
):
    # Testing the correct piecewise parameters are being assigned to each toa.
    # Operates on the piecewise_parameter_from_information_array function. Requires group_index fn to be run so we have an array of length(ntoas), filled with information on which group a toa belongs to.
    # Uses this array to apply T0X_i/A1X_i to corresponding indexes from group_index fn call. i.e. for T0X_i,T0X_j,T0X_k values and group_index return: [i,j,k] the output would be [T0X_i,T0X_j,T0X_k]
    m_piecewise = build_piecewise_model_with_two_pieces
    toa = make_toas_to_go_with_two_piece_model
    if param == "A1X":
        paramX_per_toa = m_piecewise.get_A1Xs_associated_with_toas(toa)
        test_val = [m_piecewise.A1X_0000.quantity, m_piecewise.A1X_0001.quantity]
        should_toa_reference_piecewise_parameter = [
            m_piecewise.does_toa_reference_piecewise_parameter(toa, "A1X_0000"),
            m_piecewise.does_toa_reference_piecewise_parameter(toa, "A1X_0001"),
        ]
    elif param == "T0X":
        paramX_per_toa = m_piecewise.get_T0Xs_associated_with_toas(toa)
        test_val = [m_piecewise.T0X_0000.quantity, m_piecewise.T0X_0001.quantity]
        should_toa_reference_piecewise_parameter = [
            m_piecewise.does_toa_reference_piecewise_parameter(toa, "T0X_0000"),
            m_piecewise.does_toa_reference_piecewise_parameter(toa, "T0X_0001"),
        ]

    do_toas_reference_first_piecewise_parameter = np.isclose(
        (paramX_per_toa.value - test_val[0].value), 0, atol=1e-6, rtol=0
    )
    do_toas_reference_second_piecewise_parameter = np.isclose(
        (paramX_per_toa.value - test_val[1].value), 0, atol=1e-6, rtol=0
    )
    do_toas_reference_piecewise_parameter = [
        do_toas_reference_first_piecewise_parameter,
        do_toas_reference_second_piecewise_parameter,
    ]
    np.testing.assert_array_equal(
        do_toas_reference_piecewise_parameter, should_toa_reference_piecewise_parameter
    )


# fine function
def test_problematic_group_indexes_and_ranges(model_no_pieces):
    # Test to flag issues with problematic group indexes
    # Could fold this with the next test for a mega-check exceptions are raised test
    m_piecewise = model_no_pieces
    with pytest.raises(ValueError):
        m_piecewise.add_group_range(
            m_piecewise.START.value, m_piecewise.FINISH.value, j=-1
        )
    with pytest.raises(ValueError):
        m_piecewise.add_group_range(
            m_piecewise.FINISH.value, m_piecewise.START.value, j=1
        )
    with pytest.raises(ValueError):
        m_piecewise.add_group_range(
            m_piecewise.START.value, m_piecewise.FINISH.value, j=1
        )
        m_piecewise.add_group_range(
            m_piecewise.START.value, m_piecewise.FINISH.value, j=1
        )
    with pytest.raises(ValueError):
        m_piecewise.add_group_range(
            m_piecewise.START.value, m_piecewise.FINISH.value, j=10000
        )


def test_group_index_matching(model_no_pieces):
    m_piecewise = model_no_pieces
    with pytest.raises(ValueError):
        # should flag mismatching A1 group and boundary indexes
        m_piecewise.add_group_range(
            m_piecewise.START.value, m_piecewise.FINISH.value, j=1
        )
        m_piecewise.add_piecewise_param("A1", 2, paramx=m_piecewise.A1.value)
        # Errors raised in validate, which is run when groups are "locked in"
        m_piecewise.lock_groups()
    with pytest.raises(ValueError):
        # should flag mismatching T0 group and boundary indexes
        m_piecewise.add_group_range(
            m_piecewise.START.value, m_piecewise.FINISH.value, j=1
        )
        m_piecewise.add_piecewise_param("T0", 2, paramx=m_piecewise.T0.value)
        # Errors raised in validate, which is run when groups are "locked in"
        m_piecewise.lock_groups()
    with pytest.raises(ValueError):
        # check whether boundaries are overlapping
        m_piecewise.add_group_range(55000, 55200, j=1)
        m_piecewise.add_piecewise_param("A1", 1, paramx=m_piecewise.A1.value)
        m_piecewise.add_piecewise_param("T0", 1, paramx=m_piecewise.T0.value)

        m_piecewise.add_group_range(55100, 55300, j=2)
        m_piecewise.add_piecewise_param("A1", 2, paramx=m_piecewise.A1.value)
        m_piecewise.add_piecewise_param("T0", 2, paramx=m_piecewise.T0.value)

        m_piecewise.lock_groups()
    with pytest.raises(ValueError):
        # check whether boundaries are equal
        m_piecewise.add_group_range(55000, 55000, j=1)
        m_piecewise.add_piecewise_param("A1", 1, paramx=m_piecewise.A1.value)
        m_piecewise.add_piecewise_param("T0", 1, paramx=m_piecewise.T0.value)
        m_piecewise.lock_groups()


def test_does_toa_lie_in_group_non_overlapping_complete_group_coverage(
    model_no_pieces, build_piecewise_model_with_two_pieces
):
    m_piecewise, toas = add_full_coverage_and_non_overlapping_groups_and_make_toas(
        model_no_pieces, build_piecewise_model_with_two_pieces
    )
    is_toa_in_each_group = []
    for i in range(m_piecewise.get_number_of_groups()):
        par = f"T0X_{int(i):04d}"
        is_toa_in_each_group.append(
            m_piecewise.does_toa_reference_piecewise_parameter(toas, par)
        )
    barycentric_toa = m_piecewise._parent.get_barycentric_toas(toas)
    # Logic is toas lie in group 0:10 should all be True in the piece. And False for toas out of the piece. (T0X_0000 being used)
    # and vice versa for the T0X_0001
    are_toas_within_group_boundaries_mjd_method_per_parameter = (
        return_truth_array_based_on_group_boundaries(m_piecewise, barycentric_toa)
    )
    is_toa_in_each_group = np.concatenate(
        (is_toa_in_each_group[0], is_toa_in_each_group[1])
    )
    are_toas_within_group_boundaries_mjd_method_per_parameter = np.concatenate(
        (
            are_toas_within_group_boundaries_mjd_method_per_parameter[0][0],
            are_toas_within_group_boundaries_mjd_method_per_parameter[1][0],
        )
    )
    np.testing.assert_array_equal(
        are_toas_within_group_boundaries_mjd_method_per_parameter, is_toa_in_each_group
    )


def test_does_toa_lie_in_group_incomplete_group_coverage(
    model_no_pieces, build_piecewise_model_with_two_pieces
):
    m_piecewise, toas = add_partial_coverage_groups_and_make_toas(model_no_pieces)
    is_toa_in_each_group = []
    for i in range(m_piecewise.get_number_of_groups()):
        par = f"T0X_0001"
        # should raise an error if T0X_0000 has been removed and "T0X" is effectively called in does_toa_reference_piecewise_parameter. Currently doesn't...
        is_toa_in_each_group.append(
            m_piecewise.does_toa_reference_piecewise_parameter(toas, par)
        )
    barycentric_toa = m_piecewise._parent.get_barycentric_toas(toas)
    # This test is performed to make sure toas that shouldn't be in any group are correctly being flagged.
    # Only latter half of toas should be in group. Distinct from the overlapping group test since it's not ambiguous which group they belong to since they aren't in a group
    boundaries = m_piecewise.get_group_boundaries()
    lower_edge_of_group = boundaries[0][0]
    are_toas_above_group_edge = [barycentric_toa >= lower_edge_of_group]
    do_in_piece_and_mjd_methods_of_assigning_groups_agree = np.array_equal(
        is_toa_in_each_group, are_toas_above_group_edge
    )
    assert do_in_piece_and_mjd_methods_of_assigning_groups_agree


@pytest.mark.parametrize(
    "param, index", [("T0X", 0), ("T0X", 1), ("A1X", 0), ("A1X", 1)]
)
def test_residuals_in_groups_respond_to_changes_in_corresponding_piecewise_parameter(
    model_no_pieces, build_piecewise_model_with_two_pieces, param, index
):
    m_piecewise, toas = add_full_coverage_and_non_overlapping_groups_and_make_toas(
        model_no_pieces, build_piecewise_model_with_two_pieces
    )
    rs_value = pint.residuals.Residuals(
        toas, m_piecewise, subtract_mean=False
    ).resids_value
    parameter_string = f"{param}_{int(index):04d}"
    m_piecewise_temp = add_offset_in_model_parameter(index, param, m_piecewise)
    rs_temp = pint.residuals.Residuals(
        toas, m_piecewise_temp, subtract_mean=False
    ).resids_value
    have_residuals_changed = rs_temp != rs_value
    should_residuals_change = m_piecewise.does_toa_reference_piecewise_parameter(
        toas, parameter_string
    )
    np.testing.assert_array_equal(have_residuals_changed, should_residuals_change)


@pytest.mark.parametrize(
    "param, index", [("T0X", 0), ("T0X", 1), ("A1X", 0), ("A1X", 1)]
)
# assert array equal truth_array not (~) in_piece
def test_d_delay_in_groups_respond_to_changes_in_corresponding_piecewise_parameter(
    model_no_pieces, build_piecewise_model_with_two_pieces, param, index
):
    m_piecewise, toas = add_full_coverage_and_non_overlapping_groups_and_make_toas(
        model_no_pieces, build_piecewise_model_with_two_pieces
    )
    m_piecewise_temp = add_offset_in_model_parameter(index, param, m_piecewise)
    parameter_string = f"{param}_{int(index):04d}"
    is_d_delay_changing = np.invert(
        np.isclose(
            m_piecewise_temp.d_binary_delay_d_xxxx(toas, parameter_string, None).value,
            0,
            atol=1e-11,
            rtol=0,
        )
    )
    should_d_delay_be_changing = m_piecewise.does_toa_reference_piecewise_parameter(
        toas, parameter_string
    )
    # assert toas that are in the group have some non-zero delay derivative
    np.testing.assert_array_equal(is_d_delay_changing, should_d_delay_be_changing)


def test_residuals_in_pieces_are_same_as_BT_piecewise_T0(
    model_no_pieces, model_BT, build_piecewise_model_with_one_T0_piece
):
    m_piecewise = build_piecewise_model_with_one_T0_piece
    m_non_piecewise = model_BT
    param = "T0"
    index = 0
    param_string = f"{param}X_{int(index):04d}"
    param_q = getattr(m_non_piecewise, param)
    setattr(param_q, "value", getattr(m_piecewise, param_string).value)
    toas = make_generic_toas(m_non_piecewise, 55001, 55099)
    rs_piecewise = pint.residuals.Residuals(toas, m_piecewise).time_resids
    rs_non_piecewise = pint.residuals.Residuals(toas, m_non_piecewise).time_resids
    np.testing.assert_array_equal(rs_piecewise, rs_non_piecewise)


def test_residuals_in_pieces_are_same_as_BT_piecewise_A1(
    model_no_pieces, model_BT, build_piecewise_model_with_one_A1_piece
):
    m_piecewise = build_piecewise_model_with_one_A1_piece
    m_non_piecewise = model_BT
    param = "A1"
    index = 0
    param_string = f"{param}X_{int(index):04d}"
    param_q = getattr(m_non_piecewise, param)
    setattr(param_q, "value", getattr(m_piecewise, param_string).value)
    toas = make_generic_toas(m_non_piecewise, 55001, 55099)
    rs_piecewise = pint.residuals.Residuals(toas, m_piecewise).time_resids
    rs_non_piecewise = pint.residuals.Residuals(toas, m_non_piecewise).time_resids
    np.testing.assert_allclose(rs_piecewise, rs_non_piecewise)


# --Place here for future tests--
# Same test as above but TOAs are declared outside of group boundary as well.
# Wants to check the residuals within the group are the same as those from the model used to generate them which has T0 = T0X_0000 (the only group)
# In the residual calculation there is a mean subtraction/round off occurring in changing the binary parameter/something else.
# This means: test_residuals_in_pieces_are_same_as_BT_piecewise_*, would not have exactly equal residuals.
# In fact the TOAS that are expected to replicate the BT_model residuals are systematically delayed by an amount that is larger than the noise of uniform TOAs.
# Looks like the "noise" about this extra delay matches the noise of the uniform TOAs generated by the non-piecewise model
# Suggests: Don't leave TOAs in undeclared groups, try to cover the whole date range so a TOA lies in a group until explored further
# Testing can be done if all residuals used in testing have subtract_mean and weighted_mean set to False.
# Preliminary: In all residuals calculations subtract_mean and weighted_mean set to False.
# Then implement assert_all_close with arbitrary tolerances, roughly rtol=2.5, atol=1.e-7


def test_derivatives_in_pieces_are_same_as_BT_piecewise_T0(
    model_no_pieces, model_BT, build_piecewise_model_with_one_T0_piece
):
    m_piecewise = build_piecewise_model_with_one_T0_piece
    m_non_piecewise = model_BT
    toas = make_generic_toas(m_non_piecewise, 55001, 55199)
    param = "T0"
    index = 0
    param_string = f"{param}X_{int(index):04d}"
    param_q = getattr(m_non_piecewise, param)
    setattr(param_q, "value", getattr(m_piecewise, param_string).value)
    piecewise_delays = m_piecewise.d_binary_delay_d_xxxx(
        toas, param_string, acc_delay=None
    )
    non_piecewise_delays = m_non_piecewise.d_binary_delay_d_xxxx(
        toas, param, acc_delay=None
    )
    # gets which toas that should be changing
    where_delays_should_change = m_piecewise.does_toa_reference_piecewise_parameter(
        toas, param_string
    )
    # checks the derivatives wrt T0X is the same as the derivative calc'd in the BT model for T0=T0X, for TOAs within that group
    np.testing.assert_array_equal(
        piecewise_delays[where_delays_should_change],
        non_piecewise_delays[where_delays_should_change],
    )
    # checks the derivatives wrt T0X are 0 for toas outside of the group
    np.testing.assert_array_equal(
        piecewise_delays[~where_delays_should_change],
        np.zeros(len(piecewise_delays[~where_delays_should_change])),
    )


def test_derivatives_in_pieces_are_same_as_BT_piecewise_A1(
    model_no_pieces, model_BT, build_piecewise_model_with_one_A1_piece
):
    m_piecewise = build_piecewise_model_with_one_A1_piece
    m_non_piecewise = model_BT
    toas = make_generic_toas(m_non_piecewise, 55001, 55199)
    param = "A1"
    index = 0
    param_string = f"{param}X_{int(index):04d}"
    param_q = getattr(m_non_piecewise, param)
    setattr(param_q, "value", getattr(m_piecewise, param_string).value)
    piecewise_delays = m_piecewise.d_binary_delay_d_xxxx(
        toas, param_string, acc_delay=None
    )
    non_piecewise_delays = m_non_piecewise.d_binary_delay_d_xxxx(
        toas, param, acc_delay=None
    )
    # gets which toas that should be changing
    where_delays_should_change = m_piecewise.does_toa_reference_piecewise_parameter(
        toas, param_string
    )
    # checks the derivatives wrt A1X is the same as the derivative calc'd in the BT model for A1=A1X, for TOAs within that group
    np.testing.assert_array_equal(
        piecewise_delays[where_delays_should_change],
        non_piecewise_delays[where_delays_should_change],
    )
    # checks the derivatives wrt A1X are 0 for toas outside of the group
    np.testing.assert_array_equal(
        piecewise_delays[~where_delays_should_change],
        np.zeros(len(piecewise_delays[~where_delays_should_change])),
    )


def test_multiple_piece_residuals_match_BT_model(
    build_piecewise_model_with_two_pieces, model_BT
):
    # test involving 2 pieces that cover all toas (not overlapping)
    m_piecewise = build_piecewise_model_with_two_pieces
    m_non_piecewise = model_BT
    toas = make_generic_toas(m_non_piecewise, 55001, 55199)
    rs_piecewise = pint.residuals.Residuals(
        toas, m_piecewise, subtract_mean=False, use_weighted_mean=False
    ).time_resids
    rs_preserve = pint.residuals.Residuals(
        toas, m_non_piecewise, subtract_mean=False, use_weighted_mean=False
    ).time_resids
    for index in [0, 1]:
        param_q_T0 = getattr(m_non_piecewise, "T0")
        param_q_A1 = getattr(m_non_piecewise, "A1")
        param_string_T0X = f"T0X_{int(index):04d}"
        param_string_A1X = f"A1X_{int(index):04d}"
        setattr(param_q_T0, "value", getattr(m_piecewise, param_string_T0X).value)
        setattr(param_q_A1, "value", getattr(m_piecewise, param_string_A1X).value)
        where_residuals_should_be_equal = (
            m_piecewise.does_toa_reference_piecewise_parameter(toas, param_string_T0X)
        )
        rs_non_piecewise = pint.residuals.Residuals(
            toas, m_non_piecewise, subtract_mean=False, use_weighted_mean=False
        ).time_resids
        np.testing.assert_allclose(
            rs_piecewise[where_residuals_should_be_equal],
            rs_non_piecewise[where_residuals_should_be_equal],
            atol=5e-5,
            rtol=0.005,
        )


def test_interacting_with_multiple_models(model_no_pieces):
    m_piecewise_1 = deepcopy(model_no_pieces)
    m_piecewise_2 = deepcopy(model_no_pieces)
    lower_bound = [55000, 55100.00001]
    upper_bound = [55100, 55200]
    # just check by creating the models and adding pieces we aren't adding things to the other model
    m_piecewise_1.add_group_range(lower_bound[0], upper_bound[0], j=0)
    m_piecewise_1.add_piecewise_param("T0", 0, paramx=m_piecewise_1.T0.value + 1.0e-3)
    m_piecewise_1.add_piecewise_param("A1", 0, paramx=m_piecewise_1.A1.value + 1.0e-3)
    m_piecewise_1.lock_groups()
    m_piecewise_2.add_group_range(lower_bound[1], upper_bound[1], j=0)
    m_piecewise_2.add_piecewise_param("T0", 0, paramx=m_piecewise_2.T0.value + 3.0e-3)
    m_piecewise_2.add_piecewise_param("A1", 0, paramx=m_piecewise_2.A1.value + 3.0e-3)
    m_piecewise_2.lock_groups()
    # not yet interlacing function calls, just some extra sanity checks when it comes to loading more than one model that are yet untested
    np.testing.assert_allclose(m_piecewise_1.PLB_0000.value, lower_bound[0])
    np.testing.assert_allclose(m_piecewise_1.PUB_0000.value, upper_bound[0])
    np.testing.assert_allclose(m_piecewise_2.PLB_0000.value, lower_bound[1])
    np.testing.assert_allclose(m_piecewise_2.PUB_0000.value, upper_bound[1])

    np.testing.assert_allclose(
        m_piecewise_1.T0X_0000.value, m_piecewise_1.T0.value + 1.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_1.A1X_0000.value, m_piecewise_1.A1.value + 1.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_2.T0X_0000.value, m_piecewise_2.T0.value + 3.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_2.A1X_0000.value, m_piecewise_2.A1.value + 3.0e-3
    )

    # just some arithmetic tests to see if they respond to changes for now.
    # Need to find a way of listing which parameters are updated during common function calls.
    # e.g. creating residuals, why do things like tt0 change to len(1) during the calculation?
    # This is just designed to try and confuse the reader (its following a set of stable calculations and checking they match at intervals)
    param_string_T0X = "T0X_0000"
    param_string_A1X = "A1X_0000"
    param_m1_T0 = getattr(m_piecewise_1, param_string_T0X)
    param_m1_A1 = getattr(m_piecewise_1, param_string_A1X)
    param_m2_T0 = getattr(m_piecewise_2, param_string_T0X)
    param_m2_A1 = getattr(m_piecewise_2, param_string_A1X)

    setattr(param_m1_T0, "value", getattr(m_piecewise_2, param_string_T0X).value)
    setattr(param_m1_A1, "value", getattr(m_piecewise_2, param_string_A1X).value)
    setattr(param_m2_T0, "value", getattr(m_piecewise_2, "T0").value)
    setattr(param_m2_A1, "value", getattr(m_piecewise_2, "A1").value)

    np.testing.assert_allclose(
        m_piecewise_1.T0X_0000.value, m_piecewise_1.T0.value + 3.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_1.A1X_0000.value, m_piecewise_1.A1.value + 3.0e-3
    )
    np.testing.assert_allclose(m_piecewise_2.T0X_0000.value, m_piecewise_2.T0.value)
    np.testing.assert_allclose(m_piecewise_2.A1X_0000.value, m_piecewise_2.A1.value)

    setattr(
        param_m1_T0, "value", getattr(m_piecewise_2, param_string_T0X).value + 6.0e-3
    )
    setattr(
        param_m1_A1, "value", getattr(m_piecewise_2, param_string_A1X).value + 6.0e-3
    )
    setattr(param_m2_T0, "value", getattr(m_piecewise_2, "T0").value + 3.0e-3)
    setattr(param_m2_A1, "value", getattr(m_piecewise_2, "A1").value + 3.0e-3)

    np.testing.assert_allclose(
        m_piecewise_1.T0X_0000.value, m_piecewise_1.T0.value + 6.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_1.A1X_0000.value, m_piecewise_1.A1.value + 6.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_2.T0X_0000.value, m_piecewise_2.T0.value + 3.0e-3
    )
    np.testing.assert_allclose(
        m_piecewise_2.A1X_0000.value, m_piecewise_2.A1.value + 3.0e-3
    )

    # can add more suggested tests in here
    # accepting suggestions for known nuisance operations that can result in caching
