"""Test timing model functions to test out the residuals class."""
from __future__ import absolute_import, division, print_function

import numpy as np

from pint.pulsar_mjd import Time

__all__ = ["F0"]


def F0(toa, model):

    dt = toa.get_mjds(high_precision=True) - np.array(
        Time(model.PEPOCH.value, format="pulsar_mjd", scale="utc")
    )
    # Can use dt[n].jd1 and jd2 with mpmath here if necessary
    ph = np.array([x.sec * model.F0.value for x in dt])

    return ph
