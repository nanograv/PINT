import os

import pytest
import numpy as np

from astropy import units as u
from astropy.time import Time

from pint.models import get_model, get_model_and_toas
from pint.toa import get_TOAs
import pint.fitter
from pint import simulation
from pinttestdata import datadir


@pytest.mark.parametrize("error", [1 * u.us, 1, 10 * u.ns, 1 * u.ms])
def test_fake_errors_uniform(error):
    m = get_model(os.path.join(datadir, "NGC6440E.par"))
    t = simulation.make_fake_toas_uniform(50000, 51000, 10, m, error=error)
    if isinstance(error, u.Quantity):
        assert np.all(t["error"].data * t["error"].unit == error)
    else:
        assert np.all(t["error"].data * t["error"].unit == error * u.us)


@pytest.mark.parametrize("error", [1 * u.us, 1, 10 * u.ns, 1 * u.ms])
def test_fake_errors_fromMJDs(error):
    m = get_model(os.path.join(datadir, "NGC6440E.par"))
    mjds = Time(np.arange(50000, 51000, 10), format="mjd", scale="tdb")
    t = simulation.make_fake_toas_fromMJDs(mjds, m, error=error)
    if isinstance(error, u.Quantity):
        assert np.all(t["error"].data * t["error"].unit == error)
    else:
        assert np.all(t["error"].data * t["error"].unit == error * u.us)


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


@pytest.mark.parametrize("clock", ["UTC(NIST)", "TT(BIPM2021)", "TT(BIPM2015)"])
def test_random_model_clock(clock):
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )
    m.CLOCK.value = clock
    if "BIPM" in clock:
        assert simulation.get_fake_toa_clock_versions(m)[
            "bipm_version"
        ] == m.CLOCK.value.replace("TT(", "").replace(")", "")
        assert simulation.get_fake_toa_clock_versions(m)["include_bipm"]
    else:
        assert ~simulation.get_fake_toa_clock_versions(m)["include_bipm"]
