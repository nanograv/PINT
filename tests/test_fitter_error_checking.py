#! /usr/bin/env python
import io

import numpy as np
import astropy.units as u
import pytest
import re

import pint.fitter
from pint.models import get_model
from pint.models.timing_model import MissingTOAs
from pint.toa import make_fake_toas

par_base = """
PSR J1234+5678
F0 1 0
ELAT 0 0
ELONG 0 0
PEPOCH 57000
DM 10 0
SOLARN0 0
"""


def test_dmx_no_toas():
    model = get_model(
        io.StringIO(
            "\n".join(
                [
                    par_base,
                    "DMX 15",
                    "DMX_0001 16 1",
                    "DMXR1_0001 58000",
                    "DMXR2_0001 59000",
                ]
            )
        )
    )
    toas = make_fake_toas(57000, 57900, 10, model)
    with pytest.raises(MissingTOAs) as e:
        model.validate_toas(toas)
    assert e.value.parameter_names == ["DMX_0001"]
    fitter = pint.fitter.WLSFitter(toas, model)
    with pytest.raises(MissingTOAs):
        fitter.fit_toas()


def test_jump_no_toas():
    model = get_model(io.StringIO("\n".join([par_base, "JUMP -fe L_wide 0"])))
    toas = make_fake_toas(57000, 57900, 10, model)
    assert len(model.JUMP1.select_toa_mask(toas)) == 0
    model.JUMP1.frozen = True
    model.validate_toas(toas)
    model.JUMP1.frozen = False
    with pytest.raises(ValueError):
        model.validate_toas(toas)
    model.JUMP1.frozen = False
    fitter = pint.fitter.WLSFitter(toas, model)
    with pytest.raises(ValueError):
        fitter.fit_toas()


def test_dm_barycentered():
    model = get_model(io.StringIO(par_base))
    toas = make_fake_toas(57000, 57900, 10, model, obs="@", freq=np.inf)
    model.free_params = ["F0", "DM"]
    fitter = pint.fitter.WLSFitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=".*degeneracy.*DM.*"):
        fitter.fit_toas()


@pytest.mark.parametrize("Fitter", [pint.fitter.WLSFitter, pint.fitter.GLSFitter])
def test_dmx_barycentered(Fitter):
    model = get_model(
        io.StringIO(
            "\n".join(
                [
                    par_base,
                    "DMX 15",
                    "DMX_0001 16",
                    "DMXR1_0001 58000",
                    "DMXR2_0001 59000",
                ]
            )
        )
    )
    toas = make_fake_toas(58000, 58900, 10, model, obs="@", freq=np.inf)
    model.free_params = ["F0", "DM", "DMX_0001"]
    fitter = Fitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*DM\b"):
        fitter.fit_toas()
    for p in fitter.model.free_params:
        assert not np.isnan(fitter.model[p].value)
    fitter = Fitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*DMX_0001\b"):
        fitter.fit_toas()
    for p in fitter.model.free_params:
        assert not np.isnan(fitter.model[p].value)


@pytest.mark.parametrize("Fitter", [pint.fitter.WLSFitter, pint.fitter.GLSFitter])
def test_jump_everything(Fitter):
    model = get_model(io.StringIO("\n".join([par_base, "JUMP TEL barycenter 0"])))
    toas = make_fake_toas(58000, 58900, 10, model, obs="barycenter", freq=np.inf)
    model.free_params = ["JUMP1", "F0"]
    fitter = Fitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*Offset\b"):
        fitter.fit_toas(threshold=1e-14)
    for p in fitter.model.free_params:
        assert not np.isnan(fitter.model[p].value)
    fitter = Fitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*JUMP1\b"):
        fitter.fit_toas(threshold=1e-14)
    for p in fitter.model.free_params:
        assert not np.isnan(fitter.model[p].value)


def test_jump_everything_wideband():
    model = get_model(io.StringIO("\n".join([par_base, "JUMP TEL barycenter 0"])))
    toas = make_fake_toas(58000, 58900, 10, model, obs="barycenter", freq=np.inf)
    for f in toas.table["flags"]:
        f["pp_dm"] = 15.0
        f["pp_dme"] = 1e-4
    model.free_params = ["JUMP1", "F0", "DM"]
    fitter = pint.fitter.WidebandTOAFitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*Offset\b"):
        fitter.fit_toas(threshold=1e-14)
    for p in fitter.model.free_params:
        assert not np.isnan(fitter.model[p].value)
    fitter = pint.fitter.WidebandTOAFitter(toas, model)
    with pytest.warns(pint.fitter.DegeneracyWarning, match=r".*degeneracy.*JUMP1\b"):
        fitter.fit_toas(threshold=1e-14)
    for p in fitter.model.free_params:
        assert not np.isnan(fitter.model[p].value)


@pytest.mark.parametrize("Fitter", [pint.fitter.WLSFitter, pint.fitter.GLSFitter])
def test_update_model(Fitter):
    model = get_model(io.StringIO("\n".join([par_base, "JUMP TEL barycenter 0"])))
    model.INFO.value = "-f"
    model.ECL.value = "IERS2010"
    model.TIMEEPH.value = "FB90"
    model.T2CMETHOD.value = "IERS2000B"
    toas = make_fake_toas(58000, 59000, 10, model, obs="barycenter", freq=np.inf)
    fitter = Fitter(toas, model)
    fitter.fit_toas()
    par_out = fitter.model.as_parfile()
    assert re.search(r"CLOCK *TT\(TAI\)", par_out)
    assert re.search(r"TIMEEPH *FB90", par_out)
    assert re.search(r"T2CMETHOD *IERS2000B", par_out)
    assert re.search(r"NE_SW *0.0", par_out)
    assert re.search(r"ECL *IERS2010", par_out)
    assert re.search(r"DILATEFREQ *N", par_out)
    assert re.search(r"INFO *-f", par_out)
    assert re.search(r"NTOA *10.0", par_out)
    assert re.search(r"CHI2 *\d+.\d+", par_out)
    assert re.search(r"EPHEM *DE421", par_out)
    assert re.search(r"DMDATA *0.0", par_out)
    assert re.search(r"START *58000.0", par_out)
    assert re.search(r"FINISH *59000.0", par_out)
