import io
import os
from copy import deepcopy

import numpy as np
import pytest

from hypothesis import given
from hypothesis.strategies import (
    composite,
    permutations,
)
from astropy import units as u

from pinttestdata import datadir

from pint import simulation, toa
import pint.residuals
from pint.models import get_model


@pytest.fixture(scope="session")
def model_toas_and_resids():
    parfile = os.path.join(datadir, "NGC6440E.par")
    model = get_model(parfile)
    fakes = [
        simulation.make_fake_toas_uniform(
            55000, 55500, 30, model=model, freq=1400 * u.MHz, obs="ao"
        ),
        simulation.make_fake_toas_uniform(
            55010, 55500, 40, model=model, freq=800 * u.MHz, obs="gbt"
        ),
        simulation.make_fake_toas_uniform(
            55020, 55500, 50, model=model, freq=2000 * u.MHz, obs="@"
        ),
    ]
    f = io.StringIO()
    for t in fakes:
        t.write_TOA_file(f)
    f.seek(0)
    t = toa.get_TOAs(f)
    r = pint.residuals.Residuals(t, model, subtract_mean=False)

    return model, t, r


@composite
def toas_order(draw):
    # note that draw must come before cls
    n = 120  # Number of TOAs
    ix = draw(permutations(np.arange(n)))
    return ix


@given(ix=toas_order())
def test_shuffle_toas_residuals_match(model_toas_and_resids, ix):
    model, toas, r = model_toas_and_resids
    tcopy = deepcopy(toas)
    tcopy.table = tcopy.table[ix]
    rsort = pint.residuals.Residuals(tcopy, model, subtract_mean=False)
    assert np.all(r.time_resids[ix] == rsort.time_resids)


@given(ix=toas_order())
def test_shuffle_toas_chi2_match(model_toas_and_resids, ix):
    model, toas, r = model_toas_and_resids
    tcopy = deepcopy(toas)
    tcopy.table = tcopy.table[ix]
    rsort = pint.residuals.Residuals(tcopy, model, subtract_mean=False)
    # the differences seem to be related to floating point math
    assert np.isclose(r.calc_chi2(), rsort.calc_chi2(), atol=1e-14)


@pytest.mark.parametrize("sortkey", ["freq", "mjd_float"])
def test_resorting_toas_residuals_match(sortkey, model_toas_and_resids):
    model, t, r = model_toas_and_resids
    tcopy = deepcopy(t)
    i = np.argsort(t.table[sortkey])
    tcopy.table = tcopy.table[i]
    rsort = pint.residuals.Residuals(tcopy, model, subtract_mean=False)
    assert np.all(r.time_resids[i] == rsort.time_resids)


@pytest.mark.parametrize("sortkey", ["freq", "mjd_float"])
def test_resorting_toas_chi2_match(sortkey, model_toas_and_resids):
    model, t, r = model_toas_and_resids
    tcopy = deepcopy(t)
    i = np.argsort(t.table[sortkey])
    tcopy.table = tcopy.table[i]
    rsort = pint.residuals.Residuals(tcopy, model, subtract_mean=False)
    # the differences seem to be related to floating point math
    assert np.isclose(r.calc_chi2(), rsort.calc_chi2(), atol=1e-14)


@pytest.fixture(scope="session")
def toas_datalines_and_clkcorr_and_preamble():
    shuffletoas = """FORMAT 1
        test 1234.0 54321 0 pks
        test2 888 59055 0 meerkat
        test3 350 59000 0 gbt
    """
    timfile = io.StringIO(shuffletoas)
    t = toa.get_TOAs(timfile)
    timfile.seek(0)
    lines = timfile.readlines()
    preamble = lines[0]
    # string any comments or blank lines to make sure the data lines correspond to the TOAs
    datalines = np.array(
        [
            x
            for x in lines[1:]
            if not (x.startswith("C") or x.startswith("#") or len(x.strip()) == 0)
        ]
    )
    clkcorr = t.get_flag_value("clkcorr", 0, np.float64)[0] * u.s

    return t, datalines, clkcorr, preamble


@composite
def toas_order_2(draw):
    n = 3  # Number of TOAs in shuffletoas above
    return draw(permutations(np.arange(n)))


@given(permute=toas_order_2())
def test_shuffle_toas_clock_corr(permute, toas_datalines_and_clkcorr_and_preamble):
    t, datalines, clkcorr, preamble = toas_datalines_and_clkcorr_and_preamble
    f = io.StringIO(preamble + "".join([str(x) for x in datalines[permute]]))
    t = toa.get_TOAs(f)
    clkcorr1 = t.get_flag_value("clkcorr", 0, np.float64)[0] * u.s
    assert (clkcorr1 == clkcorr[permute]).all()
