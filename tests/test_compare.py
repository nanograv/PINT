import numpy as np
import unittest
import astropy
import astropy.units as u
import pint
import pint.models as mod
import os
from copy import deepcopy as cp
from pinttestdata import datadir


class TestCompare(unittest.TestCase):
    """ Test model comparison method """

    def test_paramchange(self):

        # This test changes each parameter's magnitude by the "factor" below times the parameter's
        # uncertainty. If this factor is above the threshold sigma, warnings will appear
        # and the changed parameter's string will have an exclamation point appended. If
        # accumulate_changes is turned on, the changed model will not be reset, so each
        # iteration will also keep all previous changes/warnings.

        factor = 3.1
        threshold_sigma = 3.0
        accumulate_changes = True
        verbosities = ["max", "med", "min", "check"]

        model = mod.get_model(os.path.join(datadir, "J0613-0200_NANOGrav_9yv1.gls.par"))
        modelcp = cp(model)

        for verbosity in verbosities:
            for pn in model.params_ordered:
                if (
                    pn.startswith("DMX")
                    or pn in ["PSR", "START", "FINISH"]
                    or "EPOCH" in pn
                ):
                    continue
                param = getattr(model, pn)
                param_cp = getattr(modelcp, pn)
                if (
                    type(param_cp.quantity) in [str, astropy.time.core.Time]
                    or param_cp.quantity is None
                ):
                    continue
                if param_cp.uncertainty != None:
                    param_cp.quantity = (
                        param_cp.quantity + factor * param_cp.uncertainty
                    )
                else:
                    if type(param) == pint.models.parameter.boolParameter:
                        param.value = not param.value
                    elif param_cp.quantity != 0:
                        param_cp.quantity = 1.1 * param_cp.quantity
                    else:
                        param_cp.value += 3.0
                model.compare(
                    modelcp, threshold_sigma=threshold_sigma, verbosity=verbosity
                )
                if not accumulate_changes:
                    modelcp = cp(model)
        assert True, "Failure in parameter changing test"

    def test_uncertaintychange(self):
        # This test changes each parameter's uncertainty by the "factor" below.
        # When run, warnings will appear and the changed parameter's string
        # will have an asterisk appended. If accumulate_changes is turned on,
        # the changed model will not be reset, so each iteration will also
        # keep all previous changes/warnings.

        factor = 10
        threshold_sigma = 3.0
        accumulate_changes = True
        verbosities = ["max", "med", "min", "check"]

        model = mod.get_model(os.path.join(datadir, "J0613-0200_NANOGrav_9yv1.gls.par"))
        modelcp = cp(model)

        for verbosity in verbosities:
            for pn in model.params_ordered:
                if (
                    pn.startswith("DMX")
                    or pn in ["PSR", "START", "FINISH"]
                    or "EPOCH" in pn
                ):
                    continue
                param = getattr(model, pn)
                param_cp = getattr(modelcp, pn)
                if (
                    type(param_cp.quantity) in [str, astropy.time.core.Time]
                    or param_cp.quantity is None
                ):
                    continue
                if param_cp.uncertainty != None:
                    param_cp.uncertainty = factor * param_cp.uncertainty
                else:
                    if type(param) == pint.models.parameter.boolParameter:
                        param.value = not param.value
                    else:
                        param.uncertainty = 0 * param.units
                        param_cp.uncertainty = 3.0 * param_cp.units
                model.compare(modelcp, threshold_sigma=3.0, verbosity=verbosity)
                if not accumulate_changes:
                    modelcp = cp(model)
        assert True, "Failure in uncertainty changing test"
