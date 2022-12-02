from pint.models import get_model
from io import StringIO
import pytest


@pytest.fixture
def model():
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
        A1              0.00898636                    6.000e-08
        TASC            57277.01594431                1.100e-07
        EPS1            1.4E-6                        1.000e-05
        EPS2            1.7E-5                        9.000e-06
        OMDOT           30
        LNEDOT          0
        CLK             TT(BIPM2017)
        EPHEM           DE436
        RM              -7                            1.000e+00
        PX              1.4                           3.000e-01
        FB0             0.00017391195942              4.000e-14
        FB1             -7.7E-20                      3.000e-21
        PMELONG         3.33                          4.000e-02
        PMELAT          -1.44                         1.100e-01
        UNITS           TDB
        """
    )
    return get_model(parfile)


def test_ell1k(model):
    assert "BinaryELL1k" in model.components
    assert hasattr(model, "OMDOT") and model.OMDOT.quantity is not None
    assert hasattr(model, "LNEDOT") and model.LNEDOT.quantity is not None
