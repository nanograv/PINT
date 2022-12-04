from io import StringIO

import numpy as np
import pytest

from pint.models import get_model
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_uniform


@pytest.fixture
def model_and_toas():
    parfile = StringIO(
        """
        PSRJ            J0636+5128
        ELONG           96.363147518                  1.400e-08
        ELAT            28.24309900                   3.000e-08
        DM              11.108159
        PEPOCH          57277.00
        F0              348.5592316999902             9.000e-13
        F1              -4.1895E-16                   8.000e-20
        POSEPOCH        57277.00
        DMEPOCH         56307.00
        BINARY          ELL1k
        A1              0.00898636           1         6.000e-08
        TASC            57277.01594431       1         1.100e-07
        EPS1            1.4E-6               1         1.000e-05
        EPS2            1.7E-5               1         9.000e-06
        OMDOT           30                   1
        LNEDOT          0                    1
        CLK             TT(BIPM2017)
        EPHEM           DE436
        RM              -7                            1.000e+00
        PX              1.4                           3.000e-01
        FB0             0.00017391195942     1         4.000e-14
        FB1             -7.7E-20             1         3.000e-21
        PMELONG         3.33                          4.000e-02
        PMELAT          -1.44                         1.100e-01
        UNITS           TDB
        """
    )
    model = get_model(parfile)

    fake_toas = make_fake_toas_uniform(
        startMJD=50000, endMJD=51000, ntoas=100, model=model, add_noise=True
    )

    return model, fake_toas


def test_ell1k(model_and_toas):
    model, toas = model_and_toas
    assert "BinaryELL1k" in model.components
    assert hasattr(model, "OMDOT") and model.OMDOT.quantity is not None
    assert hasattr(model, "LNEDOT") and model.LNEDOT.quantity is not None

    res = Residuals(toas, model)
    assert np.isfinite(res.calc_chi2())


def test_ell1k_designmatrix(model_and_toas):
    model, toas = model_and_toas

    # @TODO : better tests for this
    M, _, _ = model.designmatrix(toas)
    assert np.all(np.isfinite(M))


def test_change_epoch(model_and_toas):
    model, toas = model_and_toas

    model.components["BinaryELL1k"].change_binary_epoch(55300)
