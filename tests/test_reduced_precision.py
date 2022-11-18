import numpy as np
from pint.utils import require_longdouble_precision, PINTPrecisionError
import pytest


def test_require_longdouble_precision():
    ld = np.longdouble
    np.longdouble = np.float128 = np.float64

    # There are some limited precision warnings here.
    import pint.pulsar_mjd

    with pytest.raises(PINTPrecisionError):
        require_longdouble_precision()

    np.longdouble = np.float128 = ld
