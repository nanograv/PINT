"""Utililty functions for the tests"""
import logging
import os
from io import StringIO
import unittest
import pytest
import warnings


def verify_stand_alone_binary_parameter_updates(m):
    """Verify if the stand alone binary parameters get updated.

    m: `~PINT.TimingModel` object.
        The timing model with binary
    """
    if not hasattr(m, "binary_instance"):
        warnings.warn(
            UserWarning(
                "Timing model needs to have binary components"
                " to test the stand alone binary parameters."
                " update"
            )
        )
    for binary_par in m.binary_instance.binary_params:
        standalone_par = getattr(m.binary_instance, binary_par)
        try:
            pint_par_name = m.match_param_aliases(binary_par)
        except ValueError:
            if binary_par in m.internal_params:
                pint_par_name = binary_par
            else:
                pint_par_name = None
        if pint_par_name is None:
            continue
        pint_par = getattr(m, pint_par_name)
        if pint_par.value is not None:
            if hasattr(standalone_par, "value"):
                assert pint_par.value == standalone_par.value
            else:
                assert pint_par.value == standalone_par
