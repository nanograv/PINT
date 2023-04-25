""" Test for PINT object copying"""

import os
import pytest
import numpy as np
import copy

import astropy.units as u
from astropy.table import Table
from pint.models import get_model
from pint.fitter import WidebandTOAFitter
import pint.fitter
import pint.residuals
from pint.toa import get_TOAs
from pinttestdata import datadir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")


@pytest.fixture
def model():
    return get_model(parfile)


@pytest.fixture
def toas():
    # The scope="module" setting ensures the TOAs object will be created
    # only once for the whole module, which will save time but might
    # allow accidental modifications done in one test to affect other tests.
    return get_TOAs(timfile)


def test_copy_toa_object(toas):
    toa_copy = copy.deepcopy(toas)

    assert toas == toa_copy
    assert toas is not toa_copy
    assert toas.ntoas == toa_copy.ntoas
    assert all(toas.table["mjd"] == toa_copy.table["mjd"])
    assert toas.table is not toa_copy.table
    assert toas.table[0]["flags"] is not toa_copy.table[0]["flags"]
    for c in toas.table.colnames:
        assert toas.table[c] is not toa_copy.table[c]


def test_copy_model_object(model):
    model_copy = copy.deepcopy(model)

    assert model is not model_copy
    assert len(model.params) == len(model_copy.params)
    assert (model.components.keys()) == (model_copy.components.keys())
    for p in model.params:
        assert getattr(model, p) is not getattr(model_copy, p)


def test_copy_residuals(model, toas):
    r = pint.residuals.Residuals(toas, model)
    r_copy = copy.deepcopy(r)
    assert r is not r_copy


def test_copy_fitter_object(model, toas):
    fitter = pint.fitter.Fitter(toas, model)
    fitter_copy = copy.deepcopy(fitter)
    assert fitter is not fitter_copy


def test_copy_wideband_fitter_object():
    model = get_model(os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.gls.par"))
    toas = get_TOAs(os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.tim"))
    fitter = WidebandTOAFitter([toas], model, additional_args={})
    fitter_copy = copy.deepcopy(fitter)

    assert fitter is not fitter_copy
    assert len(fitter.designmatrix_makers) == len(fitter_copy.designmatrix_makers)
    for orig, copied in zip(
        fitter.designmatrix_makers, fitter_copy.designmatrix_makers
    ):
        assert orig.derivative_quantity == copied.derivative_quantity
        assert orig.quantity_unit == copied.quantity_unit


@pytest.mark.xfail
def test_astropy_table_copy():
    """Related to https://github.com/astropy/astropy/issues/13435

    columns with dtype=object are not deep copied properly
    Xfail test as a reminder to remove relevant code from TOAs.__deepcopy__() when this is fixed in all of our dependencies
    """

    class X:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return str(self.value)

    a = np.array([1, 4, 5], dtype=np.int32)
    b = [2.0, 5.0, 8.5]
    c = ["x", "y", "z"]
    d = [10, 20, 30] * u.m / u.s
    e = [X(1), X(2), X(3)]

    t = Table(
        [a, b, c, d, e],
        names=("a", "b", "c", "d", "e"),
        meta={"name": "first table"},
    )
    t_copy = copy.deepcopy(t)

    for cname in t.colnames:
        assert t[0][cname] is not t_copy[0][cname]
