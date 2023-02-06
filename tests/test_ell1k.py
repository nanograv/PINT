from io import StringIO

import numpy as np
import pytest

from pint.models import get_model
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_uniform
from pint.fitter import DownhillWLSFitter


@pytest.fixture
def model_and_toas():
    parfile = StringIO(
        """
        # Based on J0636+5128. Only for testing. Relativistic 
        # effects may not be physically consistent.
        PSRJ            PSRTEST
        ELONG           96.363147518                  1.400e-08
        ELAT            28.24309900                   3.000e-08
        DM              11.108159
        PEPOCH          57277.00
        F0              348.5592316999902             9.000e-13
        F1              -4.1895E-16                   8.000e-20
        POSEPOCH        57277.00
        DMEPOCH         56307.00
        BINARY          ELL1k
        A1              0.0898636            1         6.000e-08
        TASC            57277.01594431       1         1.100e-07
        EPS1            1.4E-6               1         1.000e-05
        EPS2            1.7E-5               1         9.000e-06
        OMDOT           100                  1
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


@pytest.fixture
def model_and_toas_no_evol():
    parfile_ell1k = StringIO(
        """
        # Based on J0636+5128. Only for testing. Relativistic 
        # effects may not be physically consistent.
        PSRJ            PSRTEST
        ELONG           96.363147518                  1.400e-08
        ELAT            28.24309900                   3.000e-08
        DM              11.108159
        PEPOCH          57277.00
        F0              348.5592316999902             9.000e-13
        F1              -4.1895E-16                   8.000e-20
        POSEPOCH        57277.00
        DMEPOCH         56307.00
        BINARY          ELL1k
        A1              0.0898636            1         6.000e-08
        TASC            57277.01594431       1         1.100e-07
        EPS1            1.4E-6               1         1.000e-05
        EPS2            1.7E-5               1         9.000e-06
        OMDOT           0                    1
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
    model_ell1k = get_model(parfile_ell1k)

    parfile_ell1 = StringIO(
        """
        # Based on J0636+5128. Only for testing. Relativistic 
        # effects may not be physically consistent.
        PSRJ            PSRTEST
        ELONG           96.363147518                  1.400e-08
        ELAT            28.24309900                   3.000e-08
        DM              11.108159
        PEPOCH          57277.00
        F0              348.5592316999902             9.000e-13
        F1              -4.1895E-16                   8.000e-20
        POSEPOCH        57277.00
        DMEPOCH         56307.00
        BINARY          ELL1
        A1              0.0898636            1         6.000e-08
        TASC            57277.01594431       1         1.100e-07
        EPS1            1.4E-6               1         1.000e-05
        EPS2            1.7E-5               1         9.000e-06
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
    model_ell1 = get_model(parfile_ell1)

    fake_toas = make_fake_toas_uniform(
        startMJD=50000, endMJD=51000, ntoas=100, model=model_ell1k, add_noise=True
    )

    return model_ell1, model_ell1k, fake_toas


def test_ell1k(model_and_toas):
    model, toas = model_and_toas

    assert "BinaryELL1k" in model.components
    assert hasattr(model, "OMDOT") and model.OMDOT.quantity is not None
    assert hasattr(model, "LNEDOT") and model.LNEDOT.quantity is not None

    res = Residuals(toas, model)
    assert np.isfinite(res.calc_chi2())
    assert res.reduced_chi2 > 0.5 and res.reduced_chi2 < 2.0


def test_ell1k_designmatrix(model_and_toas):
    model, toas = model_and_toas

    # @TODO : better tests for this
    M, _, _ = model.designmatrix(toas)
    assert np.all(np.isfinite(M))


def test_change_epoch(model_and_toas):
    model, toas = model_and_toas

    model.components["BinaryELL1k"].change_binary_epoch(55300)


def test_fitting_omdot(model_and_toas):
    model, toas = model_and_toas

    for par in model.free_params:
        getattr(model, par).frozen = True
    model.OMDOT.frozen = False

    ftr = DownhillWLSFitter(toas, model)
    ftr.fit_toas(maxiter=5)

    assert np.isfinite(ftr.model.OMDOT.value) and np.isfinite(
        ftr.model.OMDOT.uncertainty_value
    )

    # Fitted value should be within 4-sigma
    assert (
        abs(ftr.model.OMDOT.value - model.OMDOT.value)
        / ftr.model.OMDOT.uncertainty_value
        < 4.0
    )


def test_fitting_lnedot(model_and_toas):
    model, toas = model_and_toas

    for par in model.free_params:
        getattr(model, par).frozen = True
    model.LNEDOT.frozen = False

    ftr = DownhillWLSFitter(toas, model)
    ftr.fit_toas(maxiter=5)

    assert np.isfinite(ftr.model.LNEDOT.value) and np.isfinite(
        ftr.model.LNEDOT.uncertainty_value
    )

    # Fitted value should be within 4-sigma
    assert (
        abs(ftr.model.LNEDOT.value - model.LNEDOT.value)
        / ftr.model.LNEDOT.uncertainty_value
        < 4.0
    )


def test_compare_ell1(model_and_toas_no_evol):
    model_ell1, model_ell1k, fake_toas = model_and_toas_no_evol

    # There will be a constant offset between the ELL1 delay and the ELL1k delay
    # when OMDOT=LNEDOT=0. This arises from the extra term (-3*a1*eps1)/(2*c) in
    # the Roemer delay expression.
    delay_ell1 = model_ell1.components["BinaryELL1"].binarymodel_delay(fake_toas)
    delay_ell1k = model_ell1k.components["BinaryELL1k"].binarymodel_delay(fake_toas)
    diff = delay_ell1 - delay_ell1k
    diff -= np.mean(diff)

    assert np.allclose(diff, 0)
