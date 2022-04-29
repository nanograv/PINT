import os
from copy import deepcopy

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pytest
import numpy as np
import astropy.units as u

from pint.models import get_model, get_model_and_toas
from pint.toa import get_TOAs
import pint.fitter
from pint import toa, simulation
from pinttestdata import datadir
import pint.models.parameter as param
from pint import ls
from pint import utils


@pytest.mark.parametrize(
    "fitter",
    [
        pint.fitter.GLSFitter,
        pint.fitter.WLSFitter,
        pint.fitter.DownhillWLSFitter,
        pint.fitter.DownhillGLSFitter,
    ],
)
def test_random_models(fitter):
    # Get model and TOAs
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )

    f = fitter(toas=t, model=m)
    # Do a 4-parameter fit
    f.model.free_params = ("F0", "F1", "RAJ", "DECJ")
    f.fit_toas()

    # this contains TOAs up through 54200
    # make new ones starting there
    tnew = simulation.make_fake_toas_uniform(54200, 59000, 59000 - 54200, f.model)
    dphase, mrand = simulation.calculate_random_models(f, tnew, Nmodels=30)

    # this is a bit stochastic, but I see typically < 0.14 cycles
    # for the uncertainty at 59000
    assert np.all(dphase.std(axis=0) < 0.2)

    # redo it with only F0 free
    dphase_F, mrand_F = simulation.calculate_random_models(
        f, tnew, Nmodels=100, params=["F0"]
    )

    # this should be less than the fully free version
    assert dphase_F.std(axis=0).max() < dphase.std(axis=0).max()


@pytest.mark.parametrize(
    "fitter",
    [pint.fitter.WidebandTOAFitter, pint.fitter.WidebandDownhillFitter],
)
def test_random_models_wb(fitter):
    model = get_model(os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.gls.par"))
    toas = get_TOAs(
        os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.tim"),
        ephem="DE436",
        bipm_version="BIPM2015",
    )
    f = fitter(toas, model)
    # Do a 4-parameter fit
    f.model.free_params = ("F0", "F1", "ELONG", "ELAT")
    f.fit_toas()

    tnew = simulation.make_fake_toas_uniform(
        54200, 59000, (59000 - 54200) // 10, f.model
    )
    dphase, mrand = simulation.calculate_random_models(f, tnew, Nmodels=30)

    # this is a bit stochastic, but I see typically < 1e-4 cycles for this
    # for the uncertainty at 59000
    assert np.all(dphase.std(axis=0) < 1e-4)
