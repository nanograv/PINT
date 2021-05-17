#! /usr/bin/env python
import os
from copy import deepcopy

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pytest
import numpy as np
import astropy.units as u

import pint.models as tm
from pint import fitter, toa
from pinttestdata import datadir
import pint.models.parameter as param
from pint import ls
from pint import utils


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_random_models():
    # taken from test_fitter.py/test_fitter()
    # Get model

    m = tm.get_model(os.path.join(datadir, "NGC6440E.par"))

    # Get TOAs
    t = toa.TOAs(os.path.join(datadir, "NGC6440E.tim"))
    t.apply_clock_corrections(include_bipm=False)
    t.compute_TDBs()
    try:
        planet_ephems = m.PLANET_SHAPIRO.value
    except AttributeError:
        planet_ephems = False
    t.compute_posvels(planets=planet_ephems)

    f = fitter.WLSFitter(toas=t, model=m)
    # Do a 4-parameter fit
    f.model.free_params = ("F0", "F1", "RAJ", "DECJ")
    f.fit_toas()

    # this contains TOAs up through 54200
    # make new ones starting there
    tnew = toa.make_fake_toas(54200, 59000, 59000 - 54200, f.model)
    dphase, mrand = utils.calculate_random_models(f, tnew, Nmodels=100)

    # this is a bit stochastic, but I see typically < 0.14 cycles
    # for the uncertainty at 59000
    assert np.all(dphase.std(axis=0) < 0.2)

    # redo it with only F0 free
    dphase_F, mrand_F = utils.calculate_random_models(
        f, tnew, Nmodels=100, params=["F0"]
    )

    # this should be less than the fully free version
    assert dphase_F.std(axis=0).max() < dphase.std(axis=0).max()

    # make a plot (if we can)
    dt = tnew.get_mjds() - f.model.PEPOCH.value * u.d
    plt.close()
    p1 = plt.plot(tnew.get_mjds(), dphase.std(axis=0), label="All Free")
    p2 = plt.plot(tnew.get_mjds(), dphase_F.std(axis=0), label="F0 free")
    dt = tnew.get_mjds() - f.model.PEPOCH.value * u.d
    p3 = plt.plot(
        tnew.get_mjds(),
        np.sqrt(
            (f.model.F0.uncertainty * dt) ** 2
            + (0.5 * f.model.F1.uncertainty * dt ** 2) ** 2
        ).decompose(),
        label="Analytic",
    )
    plt.xlabel("MJD")
    plt.ylabel("Phase Uncertainty (cycles)")
    plt.legend()
