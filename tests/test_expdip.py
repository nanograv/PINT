from io import StringIO
import numpy as np
import pytest
import astropy.units as u

from pint.fitter import Fitter
from pint.models.model_builder import get_model
from pint.simulation import make_fake_toas_uniform
from pint.residuals import Residuals


@pytest.fixture
def model_and_toas():
    par = """
        RAJ            05:00:00                     1
        DECJ           15:00:00                     1
        POSEPOCH       55000
        F0             100                          1
        F1             -1e-15                       1
        PEPOCH         55000
        EXPDIPEP_1     54764.272428904194001        1      
        EXPDIPAMP_1    1.6641670367524487e-06       1
        EXPDIPTAU_1    112.00425959054773           1
        EXPDIPIDX_1    -1.9148109887274356          1
        EXPEP_2        55764.272428904194001        1      
        EXPPH_2        2.6641670367524487e-06       1
        EXPTAU_2       102.00425959054773           1
        EXPINDEX_2     -2.9148109887274356          1
        EXPDIPEPS      0.01 
        TZRMJD         55000
        TZRSITE        pks
        TZRFRQ         1400
        EPHEM          DE440
        CLOCK          TT(BIPM2019)
        UNITS          TDB
    """
    m = get_model(StringIO(par))

    freqs = np.linspace(500, 1500, 8) * u.MHz
    t = make_fake_toas_uniform(
        54000,
        58000,
        4000,
        m,
        freq=freqs,
        obs="pks",
        add_noise=True,
        multi_freqs_in_epoch=True,
    )

    return m, t


def test_expdip(model_and_toas):
    m, t = model_and_toas

    assert len(m.components["SimpleExponentialDip"].get_indices()) == 2

    res = Residuals(t, m)
    assert res.reduced_chi2 < 1.5

    ftr = Fitter.auto(t, m)
    ftr.fit_toas(maxiter=10)
    assert ftr.resids.reduced_chi2 < 1.5

    for p in m.free_params:
        assert (m[p].value - ftr.model[p].value) / ftr.model[p].uncertainty_value < 3
