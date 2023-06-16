import logging
import os
import pytest

import astropy.units as u
import numpy as np

import pint.toa as toa
from pint.models import get_model
from pint.residuals import Residuals
from pinttestdata import datadir

from pint.modelutils import model_equatorial_to_ecliptic, model_ecliptic_to_equatorial


class TestEcliptic:
    """Test conversion from equatorial <-> ecliptic coordinates, and compare residuals."""

    @classmethod
    def setup_class(cls):
        # J0613 is in equatorial
        cls.parfileJ0613 = os.path.join(
            datadir, "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par"
        )
        cls.timJ0613 = os.path.join(datadir, "J0613-0200_NANOGrav_dfg+12.tim")
        cls.toasJ0613 = toa.get_TOAs(
            cls.timJ0613, ephem="DE405", planets=False, include_bipm=False
        )
        cls.modelJ0613 = get_model(cls.parfileJ0613)

        # B1855+09 is in ecliptic
        cls.parfileB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par")
        cls.timB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim")
        cls.toasB1855 = toa.get_TOAs(
            cls.timB1855, ephem="DE421", planets=False, include_bipm=False
        )
        cls.modelB1855 = get_model(cls.parfileB1855)

        cls.log = logging.getLogger("TestEcliptic")

    def test_to_ecliptic(self):
        # determine residuals with base (equatorial) model
        pint_resids = Residuals(
            self.toasJ0613, self.modelJ0613, use_weighted_mean=False
        ).time_resids.to(u.s)

        # convert model to ecliptic coordinates
        ECLmodelJ0613 = model_equatorial_to_ecliptic(self.modelJ0613)
        assert ECLmodelJ0613 is not None, "Creation of ecliptic model failed"
        assert (
            "AstrometryEcliptic" in ECLmodelJ0613.components
        ), "Creation of ecliptic model failed"
        assert (
            "AstrometryEquatorial" not in ECLmodelJ0613.components
        ), "Equatorial model still present"
        self.log.debug("Ecliptic model created")

        # determine residuals with new (ecliptic) model
        ECLpint_resids = Residuals(
            self.toasJ0613, ECLmodelJ0613, use_weighted_mean=False
        ).time_resids.to(u.s)
        self.log.debug(np.abs(pint_resids - ECLpint_resids))
        msg = (
            "Residual comparison to ecliptic model failed with max relative difference %e s"
            % np.nanmax(np.abs(pint_resids - ECLpint_resids)).value
        )
        assert np.all(np.abs(pint_resids - ECLpint_resids) < 1e-10 * u.s), msg

    def test_to_equatorial(self):
        # determine residuals with base (ecliptic) model
        pint_resids = Residuals(
            self.toasB1855, self.modelB1855, use_weighted_mean=False
        ).time_resids.to(u.s)

        # convert model to ecliptic coordinates
        EQUmodelB1855 = model_ecliptic_to_equatorial(self.modelB1855)
        assert EQUmodelB1855 is not None, "Creation of equatorial model failed"
        assert (
            "AstrometryEquatorial" in EQUmodelB1855.components
        ), "Creation of equatorial model failed"
        assert (
            "AstrometryEcliptic" not in EQUmodelB1855.components
        ), "Ecliptic model still present"
        self.log.debug("Equatorial model created")

        # determine residuals with new (equatorial) model
        EQUpint_resids = Residuals(
            self.toasB1855, EQUmodelB1855, use_weighted_mean=False
        ).time_resids.to(u.s)
        self.log.debug(np.abs(pint_resids - EQUpint_resids))
        msg = (
            "Residual comparison to ecliptic model failed with max relative difference %e s"
            % np.nanmax(np.abs(pint_resids - EQUpint_resids)).value
        )
        assert np.all(np.abs(pint_resids - EQUpint_resids) < 1e-10 * u.s), msg
