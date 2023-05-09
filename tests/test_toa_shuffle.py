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

shuffletoas = """FORMAT 1
test 1234.0 54321 0 pks
test2 888 59055 0 meerkat
test3 350 59000 0 gbt
"""


class TOAOrderSetup:
    parfile = os.path.join(datadir, "NGC6440E.par")
    model = get_model(parfile)
    # fake a multi-telescope, multi-frequency data-set and make sure the results don't depend on TOA order
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

    @classmethod
    @composite
    def toas_and_order(draw, cls):
        # note that draw must come before cls
        n = len(cls.t)
        ix = draw(permutations(np.arange(n)))
        return cls.t, ix


@given(TOAOrderSetup.toas_and_order())
def test_shuffle_toas_residuals_match(t_and_permute):
    toas, ix = t_and_permute
    tcopy = deepcopy(toas)
    tcopy.table = tcopy.table[ix]
    rsort = pint.residuals.Residuals(tcopy, TOAOrderSetup.model, subtract_mean=False)
    assert np.all(TOAOrderSetup.r.time_resids[ix] == rsort.time_resids)


@given(TOAOrderSetup.toas_and_order())
def test_shuffle_toas_chi2_match(t_and_permute):
    toas, ix = t_and_permute
    tcopy = deepcopy(toas)
    tcopy.table = tcopy.table[ix]
    rsort = pint.residuals.Residuals(tcopy, TOAOrderSetup.model, subtract_mean=False)
    # the differences seem to be related to floating point math
    assert np.isclose(TOAOrderSetup.r.calc_chi2(), rsort.calc_chi2(), atol=1e-14)


@pytest.mark.parametrize("sortkey", ["freq", "mjd_float"])
def test_resorting_toas_residuals_match(sortkey):
    tcopy = deepcopy(TOAOrderSetup.t)
    i = np.argsort(TOAOrderSetup.t.table[sortkey])
    tcopy.table = tcopy.table[i]
    rsort = pint.residuals.Residuals(tcopy, TOAOrderSetup.model, subtract_mean=False)
    assert np.all(TOAOrderSetup.r.time_resids[i] == rsort.time_resids)


@pytest.mark.parametrize("sortkey", ["freq", "mjd_float"])
def test_resorting_toas_chi2_match(sortkey):
    tcopy = deepcopy(TOAOrderSetup.t)
    i = np.argsort(TOAOrderSetup.t.table[sortkey])
    tcopy.table = tcopy.table[i]
    rsort = pint.residuals.Residuals(tcopy, TOAOrderSetup.model, subtract_mean=False)
    # the differences seem to be related to floating point math
    assert np.isclose(TOAOrderSetup.r.calc_chi2(), rsort.calc_chi2(), atol=1e-14)


class TOALineOrderSetup:
    timfile = io.StringIO(shuffletoas)
    t = toa.get_TOAs(timfile)
    timfile.seek(0)
    lines = timfile.readlines()
    preamble = lines[0]
    # string any comments or blank lines to make sure the datalines correspond to the TOAs
    datalines = np.array(
        [
            x
            for x in lines[1:]
            if not (x.startswith("C") or x.startswith("#") or len(x.strip()) == 0)
        ]
    )
    clkcorr = t.get_flag_value("clkcorr", 0, np.float64)[0] * u.s

    @classmethod
    @composite
    def toas_and_order(draw, cls):
        # note that draw must come before cls
        n = len(cls.t)
        return draw(permutations(np.arange(n)))


@given(TOALineOrderSetup.toas_and_order())
def test_shuffle_toas_clock_corr(permute):
    f = io.StringIO(
        TOALineOrderSetup.preamble
        + "".join([str(x) for x in TOALineOrderSetup.datalines[permute]])
    )
    t = toa.get_TOAs(f)
    clkcorr = t.get_flag_value("clkcorr", 0, np.float64)[0] * u.s
    assert (clkcorr == TOALineOrderSetup.clkcorr[permute]).all()
