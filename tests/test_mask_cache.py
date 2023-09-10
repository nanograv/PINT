from pinttestdata import datadir
from pint.models import get_model_and_toas
from pint.residuals import Residuals
import pytest


def test_mask_cache():
    model, toas = get_model_and_toas(
        datadir / "J1713+0747_small.gls.par", datadir / "J1713+0747_small.tim"
    )

    res1 = Residuals(toas, model)
    chi2_1 = res1.calc_chi2()

    model.lock(toas)
    res2 = Residuals(toas, model)
    chi2_2 = res2.calc_chi2()
    assert chi2_1 == chi2_2

    model.unlock()
    chi2_3 = res1.calc_chi2()
    assert chi2_1 == chi2_3

    toas.table["error"][0] = 0
    with pytest.raises(ValueError):
        model.lock(toas)
