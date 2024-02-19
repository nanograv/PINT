import pytest
import astropy
import pint
import pint.models as mod
from pint.models.parameter import funcParameter
import os
import io
from copy import deepcopy as cp
from pinttestdata import datadir


class TestCompare:
    """Test model comparison method"""

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
                    isinstance(param_cp.quantity, (str, astropy.time.core.Time))
                    or param_cp.quantity is None
                    or isinstance(param_cp, funcParameter)
                ):
                    continue
                if param_cp.uncertainty != None:
                    param_cp.quantity = (
                        param_cp.quantity + factor * param_cp.uncertainty
                    )
                elif isinstance(param, pint.models.parameter.boolParameter):
                    param.value = not param.value
                elif isinstance(param, pint.models.parameter.intParameter):
                    param.value += 1
                elif param_cp.quantity != 0:
                    param_cp.quantity = 1.1 * param_cp.quantity
                else:
                    param_cp.value += 3.0
                model.compare(
                    modelcp, threshold_sigma=threshold_sigma, verbosity=verbosity
                )
                if not accumulate_changes:
                    modelcp = cp(model)

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
                    isinstance(param_cp.quantity, (str, astropy.time.core.Time))
                    or param_cp.quantity is None
                ):
                    continue
                if param_cp.uncertainty is None:
                    if isinstance(param, pint.models.parameter.boolParameter):
                        param.value = not param.value
                    elif isinstance(param, pint.models.parameter.intParameter):
                        param.value += 1
                    else:
                        param.uncertainty = 0 * param.units
                        param_cp.uncertainty = 3.0 * param_cp.units
                else:
                    param_cp.uncertainty = factor * param_cp.uncertainty
                model.compare(modelcp, threshold_sigma=3.0, verbosity=verbosity)
                if not accumulate_changes:
                    modelcp = cp(model)

    def test_missing_uncertainties(self):
        # Removes uncertainties from both models and attempts to use compare.

        par_base1 = """
            PSR J1234+5612
            RAJ 14:34:01.00
            DECJ 56:14:00.00
            F0 1
            PEPOCH 57000
            DM 10
            DMEPOCH 57000
            DM1 2
            DMX     0.0
            DMX_0001   1.0
            DMXR1_0001     58000.0
            DMXR2_0001     58000.0
            """

        par_base2 = """
            PSR J1234+5612
            RAJ 14:34:01.00
            DECJ 56:14:00.00
            F0 1
            PEPOCH 58000
            DM 10
            DMEPOCH 57000
            DM1 2
            DMX     0.0
            DMX_0001   1.0
            DMXR1_0001     58000.0
            DMXR2_0001     58000.0
            """

        model_1 = mod.get_model(io.StringIO(par_base1))
        model_2 = mod.get_model(io.StringIO(par_base2))

        for pn in model_1.params_ordered[1:]:
            param1 = getattr(model_1, pn)
            param2 = getattr(model_2, pn)
            if (
                param1 is None
                or param2 is None
                or param1.uncertainty is None
                or param2.uncertainty is None
            ):
                continue
            param1.frozen = False
            param2.frozen = False
            param1.uncertainty = 0.1 * param1.quantity
            model_1.compare(model_2)
            model_2.compare(model_1)
            model_1 = mod.get_model(io.StringIO(par_base1))
            model_2 = mod.get_model(io.StringIO(par_base2))
