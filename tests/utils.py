"""Utililty functions for the tests"""
import logging
import os
import unittest
import warnings
from io import StringIO

import pytest


def verify_stand_alone_binary_parameter_updates(m):
    """Generic test function for binary model tests.

    Verify if the stand alone binary parameter values gets updated by the PINT
    binary model objects. This test goes through the stand-alone binary model's
    parameters and checks if the value is the same as corresponding PINT-face
    object's parameter. If the PINT-face object is unset or doesn't exist, it
    will skip the test, since some parameters in the standalone model are not
    implemented in the PINT-face object.

    Parameter
    ---------
    m: `~PINT.TimingModel` object.
        The timing model with binary.

    Note
    ----
    This test fails when the PINT object's set parameter values do not match the
    stand alone model parameter values. The stand alone model parameters should
    be updated by the .update_binary_object() when the TimingModel runs
    .setup(), .delay() and .d_delay_d_param().
    """
    if not hasattr(m, "binary_instance"):
        warnings.warn(
            UserWarning(
                "Timing model needs to have binary components"
                " to test the stand alone binary parameters."
                " update"
            )
        )
        return
    for binary_par in m.binary_instance.binary_params:
        standalone_par = getattr(m.binary_instance, binary_par)
        try:
            pint_par_name = m.match_param_aliases(binary_par)
        except ValueError:
            # The internal parameters are not in the parameter list. Thus, we
            # need to have a seperate check.
            if binary_par in m.internal_params:
                pint_par_name = binary_par
            else:
                pint_par_name = None
        if pint_par_name is None:
            continue
        pint_par = getattr(m, pint_par_name)
        if pint_par.value is not None:
            if hasattr(standalone_par, "value"):
                # Test for astropy quantity
                assert pint_par.value == standalone_par.value
            else:
                # Test for non-astropy quantity parameters.
                assert pint_par.value == standalone_par
    return
