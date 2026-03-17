import io
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals
from pint.fitter import Fitter
import pytest
import copy
import numpy as np

pars = [
    """
        RAJ         05:00:00    1
        DECJ        15:00:00    1
        POSEPOCH    55000
        VLBIAX      -0.15
        VLBIAY      0.05
        VLBIAZ      0.21
        F0          100         1
        F1          -1e-15      1
        PEPOCH      55000
        TZRMJD      55000
        TZRFRQ      1400
        TZRSITE     gbt
        UNITS       TDB
        EPHEM       DE440
        """,
    """
        ELAT        30.0        1
        ELONG       75.0        1
        POSEPOCH    55000
        VLBIAX      -0.15
        VLBIAY      0.05
        VLBIAZ      0.21
        F0          100         1
        F1          -1e-15      1
        PEPOCH      55000
        TZRMJD      55000
        TZRFRQ      1400
        TZRSITE     gbt
        UNITS       TDB
        EPHEM       DE440
        """,
]


@pytest.mark.parametrize("par", pars)
def test_vlbi_rotation(par):
    m0 = get_model(io.StringIO(par))

    assert all(param in m0 for param in ["VLBIAX", "VLBIAY", "VLBIAZ"])

    t = make_fake_toas_uniform(
        startMJD=54000,
        endMJD=56000,
        ntoas=200,
        model=m0,
        add_noise=True,
    )

    assert Residuals(t, m0).reduced_chi2 < 1.5

    ftr = Fitter.auto(t, m0)
    ftr.fit_toas()
    assert ftr.resids.reduced_chi2 < 1.5

    # ------------

    m1 = copy.deepcopy(m0)
    m1["VLBIAX"].value = m1["VLBIAY"].value = m1["VLBIAZ"].value = 0

    ftr1 = Fitter.auto(t, m1)
    ftr1.fit_toas()
    assert ftr1.resids.reduced_chi2 < 1.5

    assert np.allclose(
        ftr1.model.solar_system_geometric_delay(t), m0.solar_system_geometric_delay(t)
    )
