#! /usr/bin/env python
import io

import numpy as np
import astropy.units as u
import pytest

import pint.fitter
from pint.models import get_model
from pint.toa import make_fake_toas

par_base = """
PSR J1234+5678
F0 1 0
ELAT 0 0
ELONG 0 0
PEPOCH 57000
DM 10 0
"""


@pytest.mark.xfail(reason="DMX range checking not implemented")
def test_dmx_no_toas():
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
    toas = make_fake_toas(57000, 57900, 10, model)
    fitter = pint.fitter.WLSFitter(toas, model)
    with pytest.raises(ValueError):
        fitter.fit_toas()


def test_jump_no_toas():
    model = get_model(io.StringIO("\n".join([par_base, "JUMP -fe L_wide 0"])))
    toas = make_fake_toas(57000, 57900, 10, model)
    assert len(model.JUMP1.select_toa_mask(toas)) == 0
    model.JUMP1.frozen = True
    model.maskPar_has_toas_check(toas)
    model.JUMP1.frozen = False
    with pytest.raises(ValueError):
        model.maskPar_has_toas_check(toas)
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
        fitter.fit_toas(threshold=True)
