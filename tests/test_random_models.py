import warnings

import numpy as np
import pytest
from pinttestdata import datadir

import pint.fitter
from pint import simulation
from pint.models import get_model, get_model_and_toas
from pint.toa import get_TOAs


@pytest.mark.parametrize(
    "fitter",
    [
        pint.fitter.GLSFitter,
        pint.fitter.WLSFitter,
        pint.fitter.DownhillWLSFitter,
        pint.fitter.DownhillGLSFitter,
    ],
)
def test_random_models(fitter, pickle_dir):
    # Get model and TOAs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        m, t = get_model_and_toas(
            datadir / "NGC6440E.par",
            datadir / "NGC6440E.tim",
            picklefilename=pickle_dir,
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
def test_random_models_wb(fitter, pickle_dir):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        model = get_model(datadir / "J1614-2230_NANOGrav_12yv3.wb.gls.par")
    toas = get_TOAs(
        datadir / "J1614-2230_NANOGrav_12yv3.wb.tim",
        ephem="DE436",
        bipm_version="BIPM2015",
        picklefilename=pickle_dir,
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
