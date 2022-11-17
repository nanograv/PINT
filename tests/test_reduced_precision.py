import numpy as np

np.longdouble = np.float128 = np.float64

from pint.utils import require_longdouble_precision, PINTPrecisionError
import pytest


def test_require_longdouble_precision():
    with pytest.raises(PINTPrecisionError):
        require_longdouble_precision()
