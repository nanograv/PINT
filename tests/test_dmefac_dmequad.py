"""Test for the DM uncertainty rescaling DMEFAC and DMEQUAD
"""
from io import StringIO

import numpy as np
import pytest
import astropy.units as u

from pint.toa import get_TOAs
from pint.models.noise_model import ScaleDmError


tim = """FORMAT 1
fake 1400 54000 1.0 gbt -pp_dm 57.0 -pp_dme 0.01 -fe Rcvr_800
fake 1400 54000 1.0 ao -pp_dm 57.0 -pp_dme 0.01 -fe S_wide
fake 1400 54003 1.0 gbt -pp_dm 57.0 -pp_dme 0.01 -fe Rcvr_800
fake 1400 54004 1.0 ao -pp_dm 57.0 -pp_dme 0.01 -fe S_wide
fake 1400 54005 1.0 gbt -pp_dm 57.0 -pp_dme 0.01 -fe Rcvr_800
fake 1400 54007 1.0 vla -pp_dm 57.0 -pp_dme 0.01 -fe YUPPI
fake 1400 54008 1.0 vla -pp_dm 57.0 -pp_dme 0.01 -fe YUPPI
"""


@pytest.fixture(scope="module")
def test_toas():
    return get_TOAs(StringIO(tim), ephem="de421")


@pytest.fixture()
def test_model():
    model = ScaleDmError()
    model.setup()
    return model


def test_no_efact_noequad(test_toas, test_model):
    scale_sigma = test_model.scale_dm_sigma(test_toas)
    assert np.all(scale_sigma == 0.01 * u.pc / u.cm**3)


def test_only_one_efact(test_toas, test_model):
    test_model.DMEFAC1.key = "-fe"
    test_model.DMEFAC1.key_value = ["Rcvr_800"]
    test_model.DMEFAC1.value = 10
    test_model.setup()
    scale_sigma = test_model.scale_dm_sigma(test_toas)
    mask = test_model.DMEFAC1.select_toa_mask(test_toas)
    rest = list(set(range(test_toas.ntoas)).symmetric_difference(mask))
    assert np.all(scale_sigma[mask] == 0.1 * u.pc / u.cm**3)
    assert np.all(scale_sigma[rest] == 0.01 * u.pc / u.cm**3)


def test_only_one_equad(test_toas, test_model):
    test_model.DMEQUAD1.key = "-fe"
    test_model.DMEQUAD1.key_value = ["YUPPI"]
    test_model.DMEQUAD1.value = 10
    test_model.setup()
    scale_sigma = test_model.scale_dm_sigma(test_toas)
    mask = test_model.DMEQUAD1.select_toa_mask(test_toas)
    scaled_value = (
        np.sqrt(0.01**2 + test_model.DMEQUAD1.value**2) * u.pc / u.cm**3
    )
    rest = list(set(range(test_toas.ntoas)).symmetric_difference(mask))
    assert np.isclose(scale_sigma[mask], scaled_value).any()
    assert np.all(scale_sigma[rest] == 0.01 * u.pc / u.cm**3)


def test_only_one_equad_one_efact_same_backend(test_toas, test_model):
    test_model.DMEQUAD1.key = "-fe"
    test_model.DMEQUAD1.key_value = ["Rcvr_800"]
    test_model.DMEQUAD1.value = 10
    test_model.DMEFAC1.key = "-fe"
    test_model.DMEFAC1.key_value = ["Rcvr_800"]
    test_model.DMEFAC1.value = 10
    test_model.setup()
    scale_sigma = test_model.scale_dm_sigma(test_toas)
    mask1 = test_model.DMEQUAD1.select_toa_mask(test_toas)
    mask2 = test_model.DMEFAC1.select_toa_mask(test_toas)
    assert np.all(mask1 == mask2)
    scaled_value = (
        test_model.DMEFAC1.value
        * np.sqrt(0.01**2 + test_model.DMEQUAD1.value**2)
        * u.pc
        / u.cm**3
    )
    rest = list(set(range(test_toas.ntoas)).symmetric_difference(mask1))
    assert np.isclose(scale_sigma[mask1], scaled_value).any()
    assert np.all(scale_sigma[rest] == 0.01 * u.pc / u.cm**3)


def test_only_one_equad_one_efact_different_backend(test_toas, test_model):
    test_model.DMEQUAD1.key = "-fe"
    test_model.DMEQUAD1.key_value = ["Rcvr_800"]
    test_model.DMEQUAD1.value = 10
    test_model.DMEFAC1.key = "-fe"
    test_model.DMEFAC1.key_value = ["YUPPI"]
    test_model.DMEFAC1.value = 20
    test_model.setup()
    scale_sigma = test_model.scale_dm_sigma(test_toas)
    mask1 = test_model.DMEFAC1.select_toa_mask(test_toas)
    mask2 = test_model.DMEQUAD1.select_toa_mask(test_toas)
    scaled_value1 = test_model.DMEFAC1.value * 0.01 * u.pc / u.cm**3
    scaled_value2 = (
        np.sqrt(0.01**2 + test_model.DMEQUAD1.value**2) * u.pc / u.cm**3
    )
    assert np.isclose(scale_sigma[mask1], scaled_value1).any()
    assert np.isclose(scale_sigma[mask2], scaled_value2).any()
    rest1 = list(set(range(test_toas.ntoas)).symmetric_difference(mask1))
    rest2 = list(set(rest1).symmetric_difference(mask2))
    assert np.all(scale_sigma[rest2] == 0.01 * u.pc / u.cm**3)
