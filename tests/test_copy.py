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
from pint.toa import get_TOAs
from pinttestdata import datadir

os.chdir(datadir)


class TestObjectCopy:
    def test_copy_toa_object(self):
        toa = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")
        toa_copy = copy.deepcopy(toa)

        assert sys.getsizeof(toa) == sys.getsizeof(toa_copy)
        assert id(toa) != id(toa_copy)
        assert toa.ntoas == toa_copy.ntoas
        assert all(toa.table["mjd"] == toa_copy.table["mjd"])

    def test_copy_model_object(self):
        model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        model_copy = copy.deepcopy(model)

        assert sys.getsizeof(model) == sys.getsizeof(model_copy)
        assert id(model) != id(model_copy)
        assert len(model.params) == len(model_copy.params)
        assert list(model.components.keys()) == list(model_copy.components.keys())

    def test_copy_wideband_fitter_object(self):
        model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")
        fitter = WidebandTOAFitter([toas], model, additional_args={})
        fitter_copy = copy.deepcopy(fitter)

        assert sys.getsizeof(fitter) == sys.getsizeof(fitter_copy)
        assert id(fitter) != id(fitter_copy)
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
