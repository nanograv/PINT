import os
import pytest

import astropy.units as u
import pytest

import pint.fitter
import pint.models
import pint.residuals
import pint.toa
from pinttestdata import datadir
from pint.models.timing_model import MissingParameter
from pint import fitter

parfile = os.path.join(datadir, "piecewise.par")
parfile2 = os.path.join(datadir, "piecewise_twocomps.par")
timfile = os.path.join(datadir, "piecewise.tim")


class TestPiecewise:
    @classmethod
    def setup_class(cls):
        cls.m = pint.models.get_model(parfile)
        cls.m2 = pint.models.get_model(parfile2)
        cls.t = pint.toa.get_TOAs(timfile)

    def test_model_was_loaded(self):
        assert hasattr(self.m, "F0")
        assert hasattr(self.m, "PWF0_1")

    def test_piecewise_barytoa(self):
        # This .tim file has TOAs at the barycenter, and at infinite frequency
        rs = pint.residuals.Residuals(self.t, self.m).time_resids

        # Residuals should be less than 2.0 ms
        assert rs.std() < 2.0 * u.ms

    def test_piecewise_barytoa_2comp(self):
        # This .tim file has TOAs at the barycenter, and at infinite frequency
        rs = pint.residuals.Residuals(self.t, self.m2).time_resids

        # Residuals should be less than 2.0 ms
        assert rs.std() < 2.0 * u.ms

    def test_bad_pars(self):
        import copy

        for comp in ["PWSTART_1", "PWEP_1", "PWSTOP_1"]:
            m = copy.deepcopy(self.m)
            m.remove_param(comp)
            with pytest.raises(MissingParameter) as excinfo:
                m.validate()

            assert comp in str(excinfo.value)

    def test_fitting(self):
        f_1 = fitter.WLSFitter(toas=self.t, model=self.m)

        f_1.fit_toas()

    def test_fitting_2comp(self):
        f_1 = fitter.WLSFitter(toas=self.t, model=self.m2)

        f_1.fit_toas()
