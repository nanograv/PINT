import pytest

from pint.config import examplefile
from pint.models import get_model_and_toas


@pytest.fixture()
def data_NGC6440E():
    parfile = examplefile("NGC6440E.par.good")
    timfile = examplefile("NGC6440E.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    return model, toas


def test_param_order(data_NGC6440E):
    m, t = data_NGC6440E

    assert m.free_params == m.designmatrix(t, incoffset=False)[1]
