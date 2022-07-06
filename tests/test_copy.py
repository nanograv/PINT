""" Test for pint object copying
"""

import os
import pytest
import numpy as np
import copy
import sys

import astropy.units as u
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
    for ii in range(len(fitter.designmatrix_makers)):
        assert (
            fitter.designmatrix_makers[ii].derivative_quantity
            == fitter_copy.designmatrix_makers[ii].derivative_quantity
        )
        assert (
            fitter.designmatrix_makers[ii].quantity_unit
            == fitter_copy.designmatrix_makers[ii].quantity_unit
        )
