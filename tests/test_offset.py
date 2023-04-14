from astropy import units as u
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals
from pint.fitter import WLSFitter
from io import StringIO
import pytest


def test_offset():
    par = """
    ELAT 0.5 1
    ELONG 1.5 1
    F0 100 1
    PEPOCH 50000
    OFFSET 0.5 1
    """
    model = get_model(StringIO(par))
    assert (
        "PhaseOffset" in model.components
        and hasattr(model, "OFFSET")
        and model.OFFSET.quantity == u.Quantity(0.5, "")
    )

    toas = make_fake_toas_uniform(50000, 51000, 100, model, add_noise=True)

    with pytest.raises(ValueError):
        _ = Residuals(toas, model, subtract_mean=True)
    _ = Residuals(toas, model, subtract_mean=False)
    res = Residuals(toas, model, subtract_mean="auto")

    ftr = WLSFitter(toas, model)
    ftr.fit_toas()
    assert ftr.resids.reduced_chi2 < 2
