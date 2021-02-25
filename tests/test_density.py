import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import pint.toa as toa
from pint.models import get_model
from pylab import *
from copy import deepcopy
import io





def test_get_highest_density_range(ndays=7 * u.d):
    par_base = """
    PSR J1234+5678
    F0 1 0
    ELAT 0 0
    ELONG 0 0
    PEPOCH 57000
    DM 10 0
    SOLARN0 0
    """
    model = get_model(io.StringIO(par_base))
    toas_1 = toa.make_fake_toas(57000, 58000, 1000, model, obs="@")
    toas_2 = toa.make_fake_toas(57500, 57507, 100, model, obs="@")
    merged = toa.merge_TOAs([toas_1, toas_2])
    x1, x2 = merged.get_highest_density_range()
    x3, x4 = merged.get_highest_density_range(ndays)

    assert x1.value <= 57500 + 1e-5
    assert x1.value >= 57500 - 1e-5
    assert x3.value >= x1.value - 1e-5
    assert x3.value <= x1.value + 1e-5

    assert x2.value <= 57507 + 1e-5
    assert x2.value >= 57507 - 1e-5
    assert x4.value >= x2.value - 1e-5
    assert x4.value <= x2.value + 1e-5

