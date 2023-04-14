from astropy import units as u
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals
from pint.fitter import WLSFitter
from io import StringIO
import pytest


def test_offset():
    par = """
    F0 100 1
    OFFSET 0.5 1
    """
    m = get_model(StringIO(par))
    assert hasattr(m, "OFFSET") and m.OFFSET.quantity == u.Quantity(0.5, "rad")

    toas = make_fake_toas_uniform(50000, 51000, 100, m, add_noise=True)

    with pytest.raises(ValueError):
        res = Residuals(toas, m, subtract_mean=True)
    res = Residuals(toas, m, subtract_mean=False)

    ftr = WLSFitter(toas, m)
