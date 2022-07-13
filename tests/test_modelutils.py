import logging
import os
import warnings

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

import pint.toa as toa
from pint.models import get_model
from pint.modelutils import model_ecliptic_to_equatorial, model_equatorial_to_ecliptic
from pint.residuals import Residuals


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()

    # J0613 is in equatorial
    s.parfileJ0613 = datadir / "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par"
    s.timJ0613 = datadir / "J0613-0200_NANOGrav_dfg+12.tim"
    s.toasJ0613 = toa.get_TOAs(
        s.timJ0613,
        ephem="DE405",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        warnings.filterwarnings("ignore", message=r".*EPHVER.*")
        s.modelJ0613 = get_model(s.parfileJ0613)

    # B1855+09 is in ecliptic
    s.parfileB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par")
    s.timB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim")
    s.toasB1855 = toa.get_TOAs(
        s.timB1855,
        ephem="DE421",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    s.modelB1855 = get_model(s.parfileB1855)

    s.log = logging.getLogger("TestEcliptic")
    return s


def test_to_ecliptic(setup):
    # determine residuals with base (equatorial) model
    pint_resids = Residuals(
        setup.toasJ0613, setup.modelJ0613, use_weighted_mean=False
    ).time_resids.to(u.s)

    # convert model to ecliptic coordinates
    ECLmodelJ0613 = model_equatorial_to_ecliptic(setup.modelJ0613)
    assert ECLmodelJ0613 is not None, "Creation of ecliptic model failed"
    assert (
        "AstrometryEcliptic" in ECLmodelJ0613.components
    ), "Creation of ecliptic model failed"
    assert not (
        "AstrometryEquatorial" in ECLmodelJ0613.components
    ), "Equatorial model still present"
    setup.log.debug("Ecliptic model created")

    # determine residuals with new (ecliptic) model
    ECLpint_resids = Residuals(
        setup.toasJ0613, ECLmodelJ0613, use_weighted_mean=False
    ).time_resids.to(u.s)
    setup.log.debug(np.abs(pint_resids - ECLpint_resids))
    msg = (
        "Residual comparison to ecliptic model failed with max relative difference %e s"
        % np.nanmax(np.abs(pint_resids - ECLpint_resids)).value
    )
    assert np.all(np.abs(pint_resids - ECLpint_resids) < 1e-10 * u.s), msg


def test_to_equatorial(setup):
    # determine residuals with base (ecliptic) model
    pint_resids = Residuals(
        setup.toasB1855, setup.modelB1855, use_weighted_mean=False
    ).time_resids.to(u.s)

    # convert model to ecliptic coordinates
    EQUmodelB1855 = model_ecliptic_to_equatorial(setup.modelB1855)
    assert EQUmodelB1855 is not None, "Creation of equatorial model failed"
    assert (
        "AstrometryEquatorial" in EQUmodelB1855.components
    ), "Creation of equatorial model failed"
    assert not (
        "AstrometryEcliptic" in EQUmodelB1855.components
    ), "Ecliptic model still present"
    setup.log.debug("Equatorial model created")

    # determine residuals with new (equatorial) model
    EQUpint_resids = Residuals(
        setup.toasB1855, EQUmodelB1855, use_weighted_mean=False
    ).time_resids.to(u.s)
    setup.log.debug(np.abs(pint_resids - EQUpint_resids))
    msg = (
        "Residual comparison to ecliptic model failed with max relative difference %e s"
        % np.nanmax(np.abs(pint_resids - EQUpint_resids)).value
    )
    assert np.all(np.abs(pint_resids - EQUpint_resids) < 1e-10 * u.s), msg
