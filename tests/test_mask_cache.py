from pinttestdata import datadir
from pint.models import get_model_and_toas
from pint.residuals import Residuals


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
