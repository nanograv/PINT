from io import StringIO
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.fitter import Fitter

import pytest
import numpy as np
import astropy.units as u


@pytest.fixture
def model_and_toas():
    par = """
        RAJ         05:00:00
        DECJ        12:00:00
        F0          101.0
        F1          -1.1e-14
        PEPOCH      55000
        DM          12
        DMEPOCH     55000
        CMEPOCH     55000
        CM          3.5
        TNCHROMIDX  4.0
        EPHEM       DE440
        CLOCK       TT(BIPM2021)
        UNITS       TDB
        TZRMJD      55000
        TZRFRQ      1400
        TZRSITE     gmrt
        CMXR1_0001  53999.9
        CMXR2_0001  54500
        CMX_0001    0.5         1
        CMXR1_0002  54500.1
        CMXR2_0002  55000
        CMX_0002    -0.5        1
        CMXR1_0003  55000.1
        CMXR2_0003  55500
        CMX_0003    -0.1        1
    """

    model = get_model(StringIO(par))

    freqs = np.linspace(300, 1600, 16) * u.MHz
    toas = make_fake_toas_uniform(
        startMJD=54000,
        endMJD=55500,
        ntoas=2000,
        model=model,
        freq=freqs,
        add_noise=True,
        obs="gmrt",
        include_bipm=True,
        multi_freqs_in_epoch=True,
    )

    return model, toas


def test_cmx(model_and_toas):
    model, toas = model_and_toas

    assert "ChromaticCMX" in model.components

    ftr = Fitter.auto(toas, model)
    ftr.fit_toas()

    assert np.abs(model.CMX_0001.value - ftr.model.CMX_0001.value) / (
        3 * ftr.model.CMX_0001.uncertainty_value
    )
    assert np.abs(model.CMX_0002.value - ftr.model.CMX_0002.value) / (
        3 * ftr.model.CMX_0002.uncertainty_value
    )
    assert np.abs(model.CMX_0003.value - ftr.model.CMX_0003.value) / (
        3 * ftr.model.CMX_0003.uncertainty_value
    )

    assert ftr.resids.chi2_reduced < 1.6

    assert "CMX_0001" in str(ftr.model)
