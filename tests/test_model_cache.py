from typing import Tuple
from pint.models import get_model_and_toas
from pint.models.parameter import maskParameter
from pint.models.timing_model import TimingModel
from pint.toa import TOAs
from pinttestdata import datadir
import pytest


@pytest.fixture
def model_and_toas():
    return get_model_and_toas(
        f"{datadir}/J0023+0923_NANOGrav_11yv0.gls.par",
        f"{datadir}/J0023+0923_NANOGrav_11yv0.tim",
    )


def test_cache_set_unset(model_and_toas: Tuple[TimingModel, TOAs]):
    model, toas = model_and_toas

    assert (
        model.toas_for_cache is None
        and model.mask_cache is None
        and model.piecewise_cache is None
    )

    model._set_cache(toas)

    assert model.toas_for_cache is toas and model.mask_cache is not None

    assert all(
        p in model.mask_cache
        for p in model.params
        if model[p].quantity is not None and isinstance(model, maskParameter)
    )

    assert (
        model.piecewise_cache is not None and "DispersionDMX" in model.piecewise_cache
    )

    model._unset_cache()

    assert (
        model.toas_for_cache is None
        and model.mask_cache is None
        and model.piecewise_cache is None
    )
